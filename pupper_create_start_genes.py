#!/usr/bin/env python

def main():
    np_hidden = 0
    np_input = 1
    np_output = 2
    np_bias = 3
    nt_neuron = 0
    nt_sensor = 1
    num_inputs = 19 + 18 + 0
    num_outputs = 12
    nodes = []
    genes = []

    # next_node_id = 1
    # def get_next_node_id()
    #     nonlocal next_node_id
    #     a = next_node_id
    #     next_node_id += 1
    #     return a

    trait_num = 0
    nodes.append(f'node 1 {trait_num} {nt_sensor} {np_bias}')
    for input_id in range(2, num_inputs + 2):
        nodes.append(f'node {input_id} {trait_num} {nt_sensor} {np_input}')

    for output_id in range(num_inputs + 2, num_inputs + 2 + num_outputs):
        nodes.append(f'node {output_id} {trait_num} {nt_neuron} {np_output}')

    trait_num = 1
    innov = 0
    for input_id in range(2, num_inputs + 2):
        for output_id in range(num_inputs + 2, num_inputs + 2 + num_outputs):
            innov += 1
            genes.append(f'gene {trait_num} {input_id} {output_id} 0.0 0 {innov} 0 1')

    for node in nodes:
        print(node)
    for gene in genes:
        print(gene)


if __name__ == '__main__':
    main()
