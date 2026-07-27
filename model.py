import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import save_fig
from mRNNTorch.mRNN import mRNN

class RNNPolicy(nn.Module):
    def __init__(
            self, 
            inp_size,
            hid_size,
            output_dim, 
            activation_name="softplus",
            noise_level_act=0.01, 
            noise_level_inp=0.01, 
            constrained=False, 
            dt=10,
            t_const=100,
            batch_first=True,
            lower_bound_rec=0,
            upper_bound_rec=10,
            lower_bound_inp=0,
            upper_bound_inp=10,
            device="cpu",
            add_new_rule_inputs=False,
            num_new_inputs=10
        ):
        super().__init__()

        self.output_dim = output_dim
        self.n_layers = 1
        self.constrained = constrained
        self.device = device
        self.dt = dt
        self.t_const = t_const
        self.batch_first = batch_first
        self.sigma_recur = noise_level_act
        self.sigma_input = noise_level_inp
        self.activation_name = activation_name
        self.lower_bound_rec = lower_bound_rec
        self.upper_bound_rec = upper_bound_rec
        self.lower_bound_inp = lower_bound_inp
        self.upper_bound_inp = upper_bound_inp

        self.mrnn = mRNN(
            activation=activation_name,
            noise_level_act=noise_level_act, 
            noise_level_inp=noise_level_inp, 
            constrained=constrained, 
            dt=dt,
            tau=t_const,
            batch_first=batch_first,
            lower_bound_rec=lower_bound_rec,
            upper_bound_rec=upper_bound_rec,
            lower_bound_inp=lower_bound_inp,
            upper_bound_inp=upper_bound_inp,
            device=device
        )

        # Add Region
        self.mrnn.add_recurrent_region("region", hid_size, learnable_bias=True, device=device)
        # Add Input
        self.mrnn.add_input_region("input", inp_size, device=device)

        # Add connections
        self.mrnn.add_recurrent_connection("region", "region")
        self.mrnn.add_input_connection("input", "region")

        """
            Will replace original input region. This allows to load all other parameters but make new inputs
        """
        if add_new_rule_inputs:

            self.mrnn.add_input_region("input_new_rules", num_new_inputs, device=device)
            self.mrnn.add_input_connection("input_new_rules", "region")

            self.mrnn.add_input_region("condition", inp_size-10, device=device)
            self.mrnn.add_input_connection("condition", "region")

            self.mrnn.inp_dict["condition"].connections["region"]["parameter"].data = \
                self.mrnn.inp_dict["input"].connections["region"]["parameter"].data[:, 10:]

            # Since were just deleting make sure this is in check
            self.mrnn.total_num_inputs = inp_size-10+num_new_inputs

            del self.mrnn.inp_dict["input"]

        # Finalize connectivity
        self.mrnn.finalize_connectivity()

        self.fc = torch.nn.Linear(self.mrnn.total_num_units, output_dim)
        self.sigmoid = torch.nn.Sigmoid()

        self.to(device)

        self._rule_transfer_hook = None
        self._rule_transfer_index = None

    def forward(self, obs, x, h, *args, noise=True):
        # Forward pass through mRNN
        x, h = self.mrnn(x, obs[:, None, :], *args, noise=noise, h0=h)
        # Squeeze in the time dimension (doing timesteps one by one)
        h = h.squeeze(1)
        x = x.squeeze(1)
        # Motor output
        u = self.sigmoid(self.fc(h)).squeeze(dim=1)
        return x, h, u

    def rule_input_weight(self):
        """Return the original 28-column input matrix used by the RNN."""
        return self.mrnn.inp_dict["input"].connections["region"]["parameter"]

    def prepare_rule_column_transfer(self, rule_index=5, generator=None):
        """Freeze the policy and expose only one reinitialized rule column."""
        if not 0 <= rule_index < 10:
            raise ValueError("rule_index must identify one of the ten rule inputs")
        if "input" not in self.mrnn.inp_dict:
            raise RuntimeError("column transfer requires the original input region")

        if self._rule_transfer_hook is not None:
            self._rule_transfer_hook.remove()
            self._rule_transfer_hook = None

        for parameter in self.parameters():
            parameter.requires_grad_(False)

        input_weight = self.rule_input_weight()
        input_weight.requires_grad_(True)
        with torch.no_grad():
            torch.nn.init.xavier_normal_(
                input_weight[:, rule_index : rule_index + 1],
                generator=generator,
            )

        gradient_mask = torch.zeros_like(input_weight)
        gradient_mask[:, rule_index] = 1.0
        self._rule_transfer_hook = input_weight.register_hook(
            lambda gradient: gradient * gradient_mask
        )
        self._rule_transfer_index = rule_index
        return input_weight

    def transfer_trainable_parameters(self):
        """Return the sole parameter admitted by column-level rule transfer."""
        if self._rule_transfer_index is None:
            raise RuntimeError("prepare_rule_column_transfer must be called first")
        return (self.rule_input_weight(),)





class GRUPolicy(nn.Module):
    def __init__(
            self, 
            inp_size,
            hid_size,
            output_dim, 
            batch_first=True,
            device="cpu"
        ):
        super().__init__()

        self.output_dim = output_dim
        self.device = device
        self.batch_first = batch_first

        self.gru = nn.GRU(inp_size, hid_size, batch_first=batch_first)

        self.fc = torch.nn.Linear(hid_size, output_dim)
        self.sigmoid = torch.nn.Sigmoid()

        self.to(device)

    def forward(self, h, obs):
        # Forward pass through mRNN
        hs, h = self.gru(obs[:, None, :], h[None, :, :])
        # Squeeze in the time dimension (doing timesteps one by one)
        hs = hs.squeeze(1)
        # Motor output
        u = self.sigmoid(self.fc(hs)).squeeze(dim=1)
        return None, hs, u



class OrthogonalNet(nn.Module):
    def __init__(
            self, 
            inp_size,
            model_dict,
            activation_name="softplus",
            noise_level_act=0.01, 
            noise_level_inp=0.01, 
            constrained=False, 
            dt=10,
            t_const=100,
            batch_first=True,
            lower_bound_rec=0,
            upper_bound_rec=10,
            lower_bound_inp=0,
            upper_bound_inp=10,
            device="cpu"
        ):
        super().__init__()

        '''
            Model dict will be a dict containing the model corresponding 
            to training on a particular task
        '''

        self.n_layers = 1
        self.constrained = constrained
        self.device = device
        self.dt = dt
        self.t_const = t_const
        self.batch_first = batch_first
        self.sigma_recur = noise_level_act
        self.sigma_input = noise_level_inp
        self.activation_name = activation_name
        self.lower_bound_rec = lower_bound_rec
        self.upper_bound_rec = upper_bound_rec
        self.lower_bound_inp = lower_bound_inp
        self.upper_bound_inp = upper_bound_inp

        self.mrnn = mRNN(
            activation=activation_name,
            noise_level_act=noise_level_act, 
            noise_level_inp=noise_level_inp, 
            constrained=constrained, 
            dt=dt,
            tau=t_const,
            batch_first=batch_first,
            lower_bound_rec=lower_bound_rec,
            upper_bound_rec=upper_bound_rec,
            lower_bound_inp=lower_bound_inp,
            upper_bound_inp=upper_bound_inp,
            device=device
        )

        output_weights = []
        hidden_bias = []
        for model in model_dict:

            cur_task_model = model_dict[model]
            cur_hid_size = cur_task_model.mrnn.total_num_units
            cur_inp_size = cur_task_model.mrnn.total_num_inputs
            
            region_name = f"{model}_rec"
            inp_name = f"{model}_inp"

            # Add Region
            self.mrnn.add_recurrent_region(region_name, cur_hid_size, learnable_bias=True, device=device)

            # Add Input
            self.mrnn.add_input_region(inp_name, inp_size, device=device)

            # Add connections
            self.mrnn.add_recurrent_connection(region_name, region_name)
            self.mrnn.add_input_connection(inp_name, region_name)

            # Set the weights of these connections to those from the original model
            cur_task_model_hid_weight = cur_task_model.mrnn.region_dict["region"].connections["region"]["parameter"].data
            self.mrnn.region_dict[region_name].connections[region_name]["parameter"].data = cur_task_model_hid_weight

            cur_task_model_bias = cur_task_model.mrnn.region_dict["region"].base_firing.data
            self.mrnn.region_dict[region_name].base_firing.data = cur_task_model_bias

            # Will have to manually build output matrix
            output_weights.append(cur_task_model.fc)

        # Finalize connectivity
        self.mrnn.finalize_connectivity()

        W_rec, W_rec_mask, W_rec_sign_matrix = self.mrnn.gen_w(self.mrnn.region_dict)

        plt.imshow(W_rec.detach().numpy())
        save_fig("results/model_weights_orth.png")

        # Build output matrix here
        output_bias = []
        for out_weight in output_weights:
            output_bias.append(out_weight.bias)

        # Build output matrix here
        output_bias = []
        for out_weight in output_weights:
            output_bias.append(out_weight.bias)
        self.output_bias = torch.cat(output_bias)
        
        output_linear_weights = []
        for out_weight in output_weights:
            output_linear_weights.append(out_weight.weight)
        self.output_weight = self._block_diag(output_linear_weights)

        plt.imshow(self.output_weight.detach().numpy())
        save_fig("results/model_output_weights_orth.png")

        self.sigmoid = torch.nn.Sigmoid()

        self.to(device)
    
    # Create a block diagonal matrix
    def _block_diag(self, tensor_list):
        rows = sum(t.shape[0] for t in tensor_list)
        cols = sum(t.shape[1] for t in tensor_list)
        result = torch.zeros(rows, cols, dtype=tensor_list[0].dtype, device=tensor_list[0].device)

        r, c = 0, 0
        for t in tensor_list:
            rr, cc = t.shape
            result[r:r+rr, c:c+cc] = t
            r += rr
            c += cc

        return result

    def forward(self, obs, x, h, *args, noise=True):
        # Forward pass through mRNN
        x, h = self.mrnn(obs[:, None, :], x, h, *args, noise=noise)
        # Squeeze in the time dimension (doing timesteps one by one)
        h = h.squeeze(1)
        x = x.squeeze(1)
        # Motor output
        u = self.sigmoid((h @ self.output_weight.T) + self.output_bias).squeeze(dim=1)
        print(u.shape)
        return x, h, u
