from copy import copy

import torch
from matplotlib import pyplot as plt

from grid import Grid, Node
from utils import print_indented


class Trader:
    min_price = 1.0     # minimum possible price
    max_price = 100.0   # maximum possible price
    price_step = 1.0    # prices can be changed by this step
    # vol_step = 1.0      # vols can be changed by this step
    vol_step = 0.2      # vols can be changed by this step
    step_delay = 3      # changes are triggered when there is no trade for step_delay steps

    def __init__(self):
        self.price = 1.0
        self.max_vol = 0.0
        # self.vol = 0.0            # the volume of material that we want to trade on the current price
        self.vol = self.vol_step    # the vol is always the vol_step
        self.traded_vol = 0.0       # traded_vol <= max_vol
        self.node = None
        self.counter = 0
        self.routes = []            # sorted list of routes. First route is the worst

    def print_state(self, indent=-1):
        print_indented(indent, 'price', self.price)
        print_indented(indent, 'max_vol', self.max_vol)
        print_indented(indent, 'vol', self.vol)
        print_indented(indent, 'traded_vol', self.traded_vol)
        print_indented(indent, 'counter', self.counter)
        print_indented(indent, 'routes', len(self.routes))

    def update_price(self):
        pass

    def add_route(self, route):
        self.free_up_vol(route.vol)
        self.traded_vol += route.vol
        self.routes.append(route)
        self.routes.sort(key=self.route_score)
        # self.vol = self.max_vol - self.traded_vol   # the new vol is the remaining capacity of the trader
        # if self.vol == 0:
        #     self.vol = self.routes[0].vol   # the new vol is the vol of the worst route (that we want to replace)
        self.counter = 0

    def remove_route(self, route):
        self.routes.remove(route)
        self.traded_vol -= route.vol

    def free_up_vol(self, vol):
        """ Delete the worst routes until we have enough free vol """
        if self.traded_vol + vol <= self.max_vol:
            # there is enough vol
            return
        routes_to_delete = []
        freed_up_vol = 0.0
        for r in self.routes:
            if freed_up_vol < vol:
                routes_to_delete.append(r)
                freed_up_vol += r.vol
            else:
                break
        for r in routes_to_delete:
            r.delete()

    def route_score(self, route):
        """ Helper function to sort routes. The larger is the better """
        return 0


class Seller(Trader):
    def __init__(self):
        super().__init__()

    def add_route(self, route):
        Trader.add_route(self, route)
        route.src_price = self.price
        route.seller = self

    def update_price(self):
        if self.counter > Trader.step_delay:
            # there was no trade for step_delay turns
            if self.traded_vol < self.max_vol:
                # we need to trade more, we can go below the worst current trade
                if self.price > Trader.min_price:
                    self.price -= Trader.price_step
            else:
                # we are at full capacity, we don't want to go below the worst trade
                if self.price > self.routes[0].src_price:
                    self.price -= Trader.price_step
        else:
            # there was a recent trade
            if self.traded_vol < self.max_vol:
                # we need to trade more, don't be greedy
                pass
            else:
                # we are at full capacity, we can increase the price
                self.price += Trader.price_step
        self.counter += 1

    def route_score(self, route):
        return route.src_price


class Buyer(Trader):
    def __init__(self):
        super().__init__()

    def add_route(self, route):
        candidate = self.node.candidate
        route.vol = min(candidate.current_vol(), self.vol)
        route.transport_cost = candidate.transport_cost

        Trader.add_route(self, route)
        route.dst_price = self.price
        route.buyer = self
        route.grow(self.node)

    def update_price(self):
        if self.counter > Trader.step_delay:
            # there was no trade for step_delay turns
            if self.traded_vol < self.max_vol:
                # we still need to trade more --> increase the price
                if self.price < Trader.max_price:
                    self.price += Trader.price_step
            else:
                # we are at full capacity, we don't want to go above the worst route
                if self.routes[0].dst_price > self.price:
                    self.price += Trader.price_step
        else:
            # there was a recent trade
            if self.traded_vol < self.max_vol:
                # we still need to trade more, don't be greedy
                pass
            else:
                # we are at full capacity --> we can lower the price
                if self.price > Trader.min_price:
                    self.price -= Trader.price_step
        self.counter += 1

    def route_score(self, route):
        return -route.dst_price


class Transport:
    """ It represents an acquisition of material. Usually:
        node.transport.edge.src = node              === the destination of the material transport
        node.transport.edge.dst = neighbour_node    === the source of the material transport
    """

    def __init__(self):
        self.seller = None
        # self.src_price = 0.0        # seller's price
        # self.dst_price = 0.0        # buyer's price
        self.transported_vol = 0.0  # transported volume
        self.transport_cost = 0.0   # cost per volume
        self.edge = None            # edge pointing back to the source of the material
        self.node = None            # edge.src == node. node.candidate = this transport

    def print_state(self, indent=-1):
        # print_indented(indent, 'src_price', self.src_price)
        # print_indented(indent, 'dst_price', self.dst_price)
        print_indented(indent, 'transported_vol', self.transported_vol)
        print_indented(indent, 'transport_cost', self.transport_cost)

    def current_price(self):
        """ Price per unit, including transportation cost from the seller """
        return self.seller.price + self.transport_cost

    def current_vol(self):
        if self.seller is None:
            return 0.0
        else:
            return self.seller.vol


class NodeAlgo(Node):
    def __init__(self, id):
        super().__init__(id)
        self.buyer = None
        self.seller = None
        self.transports = []    # list of transports. This is just for visualization
        self.candidate = None   # Transport, seller candidate
        self.new_candidate = None

    def create_new_candidate(self):
        """ Finds the best transport for this node. (it can be the same node, or a neighbouring node),
            and store it as new_candidate """
        candidate = self.candidate
        self.new_candidate = candidate
        for e in self.edges:
            other_node = e.dst
            transport_cost = 0.0    # full transport cost (counting from the seller)
            partial_transport_cost = float(e.transport_cost)
            src_price = 0.0         # price at the other_node
            # vol = 0.0
            seller = None

            if partial_transport_cost > 0:
                # neighbour
                c = other_node.candidate
                if c.seller is not None:
                    src_price = c.current_price()
                    # vol = c.current_vol()
                    transport_cost = c.transport_cost + partial_transport_cost
                    seller = c.seller
            elif self.seller is not None:
                # other_node == self
                seller = self.seller
                src_price = seller.price
                # vol = seller.vol

            dst_price = src_price + partial_transport_cost  # new price candidate at self Node

            if seller is not None \
                    and (self.new_candidate.current_vol() == 0
                         or dst_price < self.new_candidate.current_price()):
                self.new_candidate = copy(candidate)
                # self.new_candidate.src_price = src_price
                # self.new_candidate.dst_price = dst_price
                # self.new_candidate.vol = vol
                self.new_candidate.edge = e
                self.new_candidate.transport_cost = transport_cost
                self.new_candidate.seller = seller

    def update_edges_from_transport(self):
        for e in self.edges:
            e.transported_material = torch.tensor(0.0)
        for t in self.transports:
            t.edge.transported_material += t.transported_vol

    def update_edges_from_candidates(self):
        for e in self.edges:
            e.transported_material = torch.tensor(0.0)
        c = self.candidate
        if c.edge is not None:
            c.edge.transported_material = torch.tensor(c.current_vol())

    def print_state(self, indent=-1):
        print_indented(indent, 'seller', self.seller)
        print_indented(indent, 'buyer', self.buyer)
        print_indented(indent, 'transports', len(self.transports))
        print_indented(indent, "candidate:")
        self.candidate.print_state(indent+1)


class Route:
    def __init__(self):
        self.transports = []
        self.seller = None
        self.buyer = None
        self.src_price = 0.0
        self.dst_price = 0.0
        self.vol = 0.0
        self.transport_cost = 0.0

    def delete(self):
        self.seller.remove_route(self)
        self.buyer.remove_route(self)
        for t in self.transports:
            t.edge.src.transports.remove(t)

    def grow(self, node: NodeAlgo):
        """ Grows the route from the buyer back to the seller """
        candidate = node.candidate  # each node already has a candidate
        # candidate.vol -= self.vol   # decrease the candidate. (it might be still > 0)
        edge = candidate.edge       # edge pointing to the source of the material
        src_node = edge.dst         # 'node' receives the material from src_node
        t = copy(candidate)         # this will be the actual transport
        t.transported_vol = self.vol            # the other params of t are already set
        self.transports.append(t)
        node.transports.append(t)   # the transport is always added to the src node, not the dst
        if node == src_node:
            # t is at the seller, this is the end of the route
            node.seller.add_route(self)
        else:
            # continue the route
            self.grow(src_node)

    def print_state(self, indent=-1):
        print_indented(indent, 'src_price', self.src_price)
        print_indented(indent, 'dst_price', self.dst_price)
        print_indented(indent, 'vol', self.vol)
        print_indented(indent, "transport_cost:", self.transport_cost)
        print_indented(indent, 'transports', len(self.transports))


class GridAlgo(Grid):
    def __init__(self, row_count, col_count, prod_count=-1):
        super().__init__(row_count, col_count, prod_count)
        self.sellers = []
        self.buyers = []
        # self.routes = []
        for n in self.nodes:
            n.candidate = Transport()
            if n.influx > 0:
                seller = Seller()
                seller.max_vol = float(n.influx)
                seller.node = n
                n.seller = seller
                self.sellers.append(seller)
            elif n.influx < 0:
                buyer = Buyer()
                buyer.max_vol = float(-n.influx)
                buyer.node = n
                n.buyer = buyer
                self.buyers.append(buyer)

    def _create_node(self, id):
        return NodeAlgo(id)

    def create_routes(self):
        for b in self.buyers:
            node = b.node
            c = node.candidate
            if c.current_vol() > 0 and c.current_price() < b.price:
                # seller price < buyer price
                route = Route()
                b.add_route(route)

    def update_edges(self):
        for n in self.nodes:
            n.update_edges_from_transport()
            # n.update_edges_from_candidates()

    def update_candidates(self):
        for n in self.nodes:
            n.create_new_candidate()
        for n in self.nodes:
            n.candidate = n.new_candidate

    def update_prices(self):
        for t in (self.sellers + self.buyers):
            t.update_price()

    def run(self):
        for i in range(1000):
            print("step", i)
            self.update_candidates()
            self.create_routes()
            self.update_prices()
            self.update_edges()
            self.draw_state()
            # self.print_state()
        plt.ioff()
        plt.show()

    def print_state(self):
        print("=====================================================================")
        print("sellers:")
        for s in self.sellers:
            s.print_state(1)
            print("")
        print("buyers:")
        for b in self.buyers:
            b.print_state(1)
            print("")
        print("nodes:")
        for n in self.nodes:
            n.print_state(1)
            print("")
        print("routes:")
        for s in self.sellers:
            for r in s.routes:
                r.print_state(1)
                print("")


if __name__ == "__main__":
    size = 10
    grid = GridAlgo(size, size, 4)
    grid.run()
