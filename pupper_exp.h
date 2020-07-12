#ifndef NEAT__PUPPER_EXP_H
#define NEAT__PUPPER_EXP_H

#include "neat.h"
#include "network.h"
#include "population.h"
#include "organism.h"
#include "genome.h"
#include "species.h"
#include "mujoco.h"
#include "glfw3.h"

using namespace std;
using namespace NEAT;

class MjVis {

public:
  MjVis(mjModel *mj_model, mjData *mj_data);
  ~MjVis();

  void draw();

private:
  mjModel *mj_model;
  mjData *mj_data;
  mjvCamera cam;                      // abstract camera
  mjvOption opt;                      // visualization options
  mjvScene scn;                       // abstract scene
  mjrContext con;                     // custom GPU context
  GLFWwindow* window;
};

Population *pupper_simulate(int gens);
bool pupper_evaluate(Organism *org, mjModel *mj_model, mjData *mj_data, MjVis *mj_vsi);
int pupper_generation(
    Population *pop, mjModel *mj_model, mjData *mj_data, MjVis *mj_vsi,
    int generation, const char *filename, int &winner_num, int &winner_genes, int &winner_nodes);

#endif
