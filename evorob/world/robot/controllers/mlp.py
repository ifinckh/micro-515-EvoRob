import numpy as np

from evorob.world.robot.controllers.base import Controller


class NeuralNetworkController(Controller):
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 16):
        self.n_input = input_size
        self.n_output = output_size
        self.n_hidden = hidden_size

        self.input_to_hidden = np.random.uniform(-1, 1, (hidden_size, input_size))
        self.hidden_to_output = np.random.uniform(-1, 1, (output_size, hidden_size))
        # biases for hidden and output layers
        self.bias_hidden = np.random.uniform(-1, 1, (hidden_size,))
        self.bias_output = np.random.uniform(-1, 1, (output_size,))

        self.n_params_i2h = input_size * hidden_size
        self.n_params_h2o = hidden_size * output_size   
        self.n_params_bh = hidden_size
        self.n_params_bo = output_size

        self.n_params = self.get_num_params()

    def get_action(self, state):
        hidden = np.tanh(state @ self.input_to_hidden.T + self.bias_hidden)
        output = np.tanh(hidden @ self.hidden_to_output.T + self.bias_output)
        return np.clip(output, -1, 1)

    def set_weights(self, encoding):
        if encoding.size != self.get_num_params():
            raise ValueError(
                f"Encoding length {encoding.size} does not match expected {self.get_num_params()} "
                f"(i2h={self.n_params_i2h}, h2o={self.n_params_h2o}, bh={self.n_params_bh}, bo={self.n_params_bo})"
            )

        self.input_to_hidden = encoding[:self.n_params_i2h].reshape(self.n_hidden, self.n_input)
        start = self.n_params_i2h
        end = start + self.n_params_h2o
        self.hidden_to_output = encoding[start:end].reshape(self.n_output, self.n_hidden)

        start = end
        end = start + self.n_params_bh
        self.bias_hidden = encoding[start:end].reshape(self.n_hidden,)

        start = end
        end = start + self.n_params_bo
        self.bias_output = encoding[start:end].reshape(self.n_output,)

    def geno2pheno(self, genotype):
        self.set_weights(genotype)

    def get_num_params(self):
        return self.n_params_i2h + self.n_params_h2o + self.n_params_bh + self.n_params_bo

    def reset_controller(self, batch_size=1) -> None:
        pass
