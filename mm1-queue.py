import numpy as np
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class QueueState:
    clock: float= 0.0
    queue_length: int = 0
    cum_waiting_time: float = .0
    queue_vector : List[float] = None
    num_customers_served: int = 0
    queue_length_history : List[tuple] = None

    def __post_init__(self):
        if self.queue_vector is None:
            self.queue_vector = []
        if self.queue_length_history is None:
            self.queue_length_history = []

class MM1Simulator:
    def __init__(self, arrival_rate:float, service_rate:float, horizon: float):
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.horizon = horizon
        self.state = QueueState()

        self.next_arrival = self.generate_exponential(self.arrival_rate)
        self.next_departure = np.inf

    def generate_exponential(self, rate:float) -> float:
        return np.random.exponential(1/rate)
        
    def handle_arrival(self):
        self.state.clock = self.next_arrival
        self.state.queue_length += 1
        self.state.queue_vector.append(self.next_arrival)
        self.state.queue_length_history.append((self.state.clock,
                                                self.state.queue_length))
        

        self.next_arrival = self.state.clock + self.generate_exponential(self.arrival_rate)

        if self.state.queue_length == 1:
            self.next_departure = self.state.clock + self.generate_exponential(self.service_rate)

    def handle_departure(self):
        self.state.clock = self.next_departure

        self.state.num_customers_served  += 1
        self.state.cum_waiting_time += (self.state.clock - 
                                        self.state.queue_vector[0])
        self.state.queue_length -= 1
        self.state.queue_vector.pop(0)
        

        self.state.queue_length_history.append((self.state.clock,
                                                self.state.queue_length))

        if self.state.queue_length >= 1:
            self.next_departure = self.state.clock + self.generate_exponential(self.service_rate)
        else:
            self.next_departure = np.inf

    def run(self):
        last_update = 0

        while min(self.next_arrival, self.next_departure) < self.horizon:
            if self.next_arrival < self.next_departure:
                self.handle_arrival()
            else:
                self.handle_departure()

            current_time = self.state.clock
            
            if current_time > last_update:
                last_update = current_time
                #print(self.state.clock, self.state.queue_length, self.state.queue_length_history, self.state.queue_vector)
        
        if self.state.num_customers_served == 0:
            return 0.0
        
        return self.state.cum_waiting_time / self.state.num_customers_served
        
def main():
    arrival_rate = 3.0
    service_rate = 1.5

    horizon = 10
    simulator = MM1Simulator(arrival_rate, service_rate, horizon)
    print("Average waiting time:", simulator.run())

if __name__ == "__main__":
    main()
                


