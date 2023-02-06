import torch


class World:
    def __init__(self, grid, model):
        self.grid = grid
        self.model = model


    """
        - apply model (n times):
            (a_b, flux_b, weight) = model(a, b), (b_a, flux_a) = model(b, a)
            a = [const_influx, ext1, ext2, ... extn]
            weight: used in the next step (see below)
            influx: production/consumption of the material on the a cell
            flux_b: used to determine the a->b flux
        - normalize cell states:
            a = influx , a_b[0] * a_b + a_c[0] * a_c + ...
            use softmax to compute the actual a->b flux from a(flux_b), a(flux_c)... values 
        - move material (n times)
            move material from influx cells, compute fuel cost, and measure reward in outflux cells.
            Ma' = Mb * flux_ba + Mc * flux_ca + ...
            fuel cost of the transport:
            Fa' = Dab * Mb * flux_ba * Fb + ... // Dab = distance of ab. It can be 1 or sqrt(2)        
            Reward_a = Ma - Fa 
        - backpropagate using the reward
        
    """

    def run(self):
        while True:
            new_data = []
            for a in self.grid.nodes:
                sum_w = 0
                sum = torch.zeros(a.data.size - 1)
                fluxes = torch.Tensor(len(a.edges))
                for i, b in enumerate(a.edges):
                    ab, fluxes[i], weight = self.model(a, b)
                    sum += weight * ab
                    sum_w += weight
                a.fluxes = torch.softmax(fluxes)
                new_data.append(sum / sum_w)


