#ifndef NEAT__PUPPER_EXP_H
#define NEAT__PUPPER_EXP_H

#include "neat.h"
#include "network.h"
#include "population.h"
#include "organism.h"
#include "genome.h"
#include "species.h"

using namespace std;
using namespace NEAT;

Population *pupper_simulate(int gens);
bool pupper_evaluate(Organism *org);
int pupper_epoch(Population *pop, int generation, char *filename, int &winner_num, int &winner_genes, int &winner_nodes);

#endif
