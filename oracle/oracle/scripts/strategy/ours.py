

# 需要

class MyStrategy:
    def __init__(self, moe_model, budget):
        self.moe_model = moe_model
        self.expert_in_memory = [[0 for _ in range(self.moe_model.expert_num) ] for _ in range(self.moe_model.layer_num)]
        self.predictor = self.init_predictor()

        self.budget = budget

    def get_action(self, obs):
        return 0

    def init_memory(self, ):
        # load none-exeprt params

        pass

    def infer(self, x):
        x = self.moe_model.infer(x)
        routing_predicts = self.predictor.predict(x)
        self.load_experts(routing_predicts)
        # load needed expertss

        for i in range(len(routing_predicts)):
            pass

    def load_experts(self, layer, expert_idices):
        toload = expert_idices[:self.budget]
        # load layer i
        for j in toload:
            if self.expert_in_memory[layer][j] == 0:
                # load expert
                pass
            else:
                print(f"Expert {layer}{j} already loaded")
        pass
        # load experts

    def init_predictor(self,):
        if True:
            # load predictor from file
            pass
        else:
            print("No predictor found, using random predictor")
            pass