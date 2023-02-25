import msvcrt

import torch
import torch.nn.functional as F

from grid import Grid


class GridNN(Grid):
    def __init__(self, row_count, col_count, prod_count, model, optimizer):
        super().__init__(row_count, col_count, prod_count)
        self.model = None
        self.optimizer = None
        self.model_iteration_count = 3 * (row_count + col_count - 2)
        self.transport_iteration_count = row_count + col_count - 2
        self.consumed_material = torch.tensor(0.0)
        self.fuel_cost = torch.tensor(0.0)
        self.set_model(model, optimizer)

    def set_model(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        for node in self.nodes:
            node.data = torch.zeros(self.model.node_features_count)
            node.material = torch.tensor(0.0)
            for edge in node.edges:
                edge.data = torch.zeros(self.model.edge_features_count)

    def _model_iterations(self):
        for time in range(self.model_iteration_count):
            self._model_iteration()

    def _transport_iterations(self):
        self._produce_material()
        for time in range(self.transport_iteration_count):
            self._transport_iteration()
        # self._consume_material()

    def _model_iteration(self):
        data_size = self.nodes[0].data.size(0)
        for a in self.nodes:
            a.weights = torch.tensor(0.0)
            a.new_data = torch.zeros(data_size)

        for a in self.nodes:
            fluxes = torch.zeros(len(a.edges))
            for i, e in enumerate(a.edges):
                b = e.dst
                da, db, e.data = self.model(a, e)
                fluxes[i] = F.relu(e.data[0])
                # weight_a = F.relu(da[0])
                # weight_b = F.relu(db[0])
                # a.new_data += da * weight_a
                # b.new_data += db * weight_b
                # a.weights += weight_a
                # b.weights += weight_b
                a.new_data = torch.maximum(a.new_data, da)
                b.new_data = torch.maximum(b.new_data, db)

            fluxes = F.softmax(fluxes, dim=0)
            for i, e in enumerate(a.edges):
                e.data[0] = fluxes[i]

        for a in self.nodes:
            a.data = a.new_data  # / torch.clamp(a.weights, min=0.000001)

    def _model_iteration2(self):
        data_size = self.nodes[0].data.size(0)
        for a in self.nodes:
            a.weights = torch.tensor(0.0)
            a.new_data = torch.zeros(data_size)

        for a in self.nodes:
            fluxes = torch.zeros(len(a.edges))
            for i, e in enumerate(a.edges):
                b = e.dst
                db, e.data = self.model(a, e)
                fluxes[i] = F.relu(e.data[0])
                # b.new_data = torch.maximum(b.new_data, db)
                weight_b = F.relu(db[0])
                b.new_data += db * weight_b
                b.weights += weight_b

            fluxes = F.softmax(fluxes, dim=0)
            for i, e in enumerate(a.edges):
                e.data[0]\
                    = fluxes[i]

        for a in self.nodes:
            a.data = a.new_data / torch.clamp(a.weights, min=0.000001)

    def _transport_iteration(self):
        for a in self.nodes:
            a.new_material = torch.tensor(0.0)
            a.new_fuel_cost = torch.tensor(0.0)
        for a in self.nodes:
            for e in a.edges:
                b = e.dst
                transported_material = e.data[0] * a.material
                if a == b:
                    new_full_consumed = torch.minimum(a.consumed_material + transported_material, F.relu(-a.influx))
                    consumed = new_full_consumed - a.consumed_material
                    transported_material = transported_material - consumed
                    a.consumed_material = new_full_consumed
                    self.consumed_material = self.consumed_material + consumed
                else:
                    b.new_material += transported_material

                e.transported_material += transported_material
                fuel_cost = transported_material * (e.transport_cost + a.fuel_cost)
                b.new_fuel_cost += fuel_cost
                self.fuel_cost += fuel_cost
        for i, a in enumerate(self.nodes):
            a.material = a.new_material
            a.fuel_cost = a.new_fuel_cost

    def _produce_material(self):
        for a in self.nodes:
            a.material += F.relu(a.influx)

    def _consume_material(self):
        for a in self.nodes:
            consumed = torch.minimum(a.material, F.relu(-a.influx))
            # self.fuel_cost = self.fuel_cost + a.fuel_cost * consumed / a.material
            a.material = a.material - consumed
            a.consumed_material += consumed
            self.consumed_material = self.consumed_material + consumed

    def forward(self):
        self._model_iterations()
        self._transport_iterations()

    def run(self, backprop, print_stats):
        with torch.autograd.set_detect_anomaly(False):
            for i in range(1):
                self.consumed_material = torch.tensor(0.0)
                self.fuel_cost = torch.tensor(0.0)
                self.optimizer.zero_grad()
                self.model.train()
                self()

                # print(f"reward: {100 * self.consumed_material.item() / self.max_reward}%")

                loss = - (self.consumed_material - 0.1 * self.fuel_cost)

                if print_stats:
                    print(f"consumed: {100 * self.consumed_material.item() / self.max_reward:.2f}%"
                          f",  fuel_cost: {self.fuel_cost:.2f},  reward: {-loss:.2f}")
                    self.draw_state()

                if self.optimizer.loss is None:
                    self.optimizer.loss = loss
                else:
                    self.optimizer.loss += loss

                if backprop:
                    self.optimizer.loss.backward()
                    self.optimizer.step()
                    self.optimizer.loss = None

                full_material = 0.0
                for a in self.nodes:
                    full_material += a.material
                    a.material = a.material.detach()
                    a.data = a.data.detach()

                    #if a.influx < 0:
                    #    print(f'consumed material: {a.consumed_material.item()}')
                    a.consumed_material = torch.tensor(0.0)
                    a.material = torch.tensor(0.0)
                    a.fuel_cost = torch.tensor(0.0)
                    a.data = torch.zeros(self.model.node_features_count)
                    for e in a.edges:
                        e.data = e.data.detach()
                        e.data = torch.zeros(self.model.edge_features_count)
                        e.transported_material = torch.tensor(0.0)

                # print(f'full material: {full_material}')

                if msvcrt.kbhit():
                    print("you pressed", msvcrt.getch(), "so now i will quit")
        # torch.save(self.model.state_dict(), 'SimCell.p')