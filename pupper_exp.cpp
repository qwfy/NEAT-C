#include "experiments.h"
#include "pupper_exp.h"
#include "mujoco.h"
#include "glfw3.h"
#include <cstring>
#include <string>
#include <tuple>
#include <vector>
#include <cmath>

#define VIS_EACH

Population *pupper_simulate(int gens) {
  Population *pop = nullptr;
  Genome *start_genome;
  char _param_name[20];
  int id;

  ostringstream *file_name_buf;
  int gen;

  // Hold records for each run
  int evals[NEAT::num_runs];
  int genes[NEAT::num_runs];
  int nodes[NEAT::num_runs];
  int winner_num;
  int winner_genes;
  int winner_nodes;

  // For averaging
  int total_evals = 0;
  int total_genes = 0;
  int total_nodes = 0;
  int run_count;
  int samples;

  memset(evals, 0, NEAT::num_runs*sizeof(int));
  memset(genes, 0, NEAT::num_runs*sizeof(int));
  memset(nodes, 0, NEAT::num_runs*sizeof(int));

  ifstream input_file("pupper_start_genes.txt", ios::in);
  if (!input_file.is_open()) {
    cerr << "Failed to open start genes" << endl;
    return nullptr;
  }

  cout << "Reading in the start genome" << endl;
  input_file >> _param_name;
  input_file >> id;
  cout << "Reading in Genome id " << id << endl;
  start_genome = new Genome(id, input_file);
  input_file.close();

  int mj_error_size = 1024;
  char mj_error[mj_error_size];

  string mj_license_path = "/home/incomplete/.mujoco/mjkey.txt";
  string mj_model_path = "/home/incomplete/ai/pupper/StanfordQuadruped/sim/pupper_mujoco.xml";

  // activate MuJoCo
  mj_activate(mj_license_path.c_str());

  // load model from file and check for errors
  mjModel *mj_model = mj_loadXML(mj_model_path.c_str(), nullptr, mj_error, mj_error_size);
  if (!mj_model) {
    cerr << "Failed to load mujoco mj_model " << mj_model_path << ": " << mj_error << endl;
    return nullptr;
  }

  // make data corresponding to mj_model
  mjData *mj_data = mj_makeData(mj_model);

  MjVis *mj_vis = nullptr;
#ifdef VIS_EACH
  mj_vis = new MjVis(mj_model, mj_data);
#endif


  for (run_count = 0; run_count < NEAT::num_runs; run_count++) {
    // Spawn the Population
    cout << "Spawning Population off Genome" << endl;

    pop = new Population(start_genome, NEAT::pop_size);

    cout << "Verifying Spawned Pop" << endl;
    pop->verify();

    for (gen = 1; gen <= gens; gen++) {
      cout << "Generation " << gen << endl;

      file_name_buf = new ostringstream();
      (*file_name_buf) << "pupper_out/gen_" << gen << ends;
      string filename = file_name_buf->str();

      // Check for success
      if (pupper_generation(
            pop, mj_model, mj_data, mj_vis,
            gen, filename.c_str(), winner_num, winner_genes, winner_nodes)) {
        // Collect Stats on end of experiment
        evals[run_count] = NEAT::pop_size*(gen - 1) + winner_num;
        genes[run_count] = winner_genes;
        nodes[run_count] = winner_nodes;
        gen = gens;
      }

      // Clear output filename
      file_name_buf->clear();
      delete file_name_buf;
    }

    if (run_count < NEAT::num_runs - 1) {
      delete pop;
    }
  }

  // Average and print stats
  cout << "Nodes: " << endl;
  for (run_count = 0; run_count < NEAT::num_runs; run_count++) {
    cout << nodes[run_count] << endl;
    total_nodes += nodes[run_count];
  }

  cout << "Genes: " << endl;
  for (run_count = 0; run_count < NEAT::num_runs; run_count++) {
    cout << genes[run_count] << endl;
    total_genes += genes[run_count];
  }

  cout << "Evals " << endl;
  samples = 0;
  for (run_count = 0; run_count < NEAT::num_runs; run_count++) {
    cout << evals[run_count] << endl;
    if (evals[run_count] > 0) {
      total_evals += evals[run_count];
      samples++;
    }
  }

  cout << "Failures: " << (NEAT::num_runs - samples) << " out of " << NEAT::num_runs << " runs" << endl;
  cout << "Average Nodes: " << (samples > 0 ? (double) total_nodes/samples : 0) << endl;
  cout << "Average Genes: " << (samples > 0 ? (double) total_genes/samples : 0) << endl;
  cout << "Average Evals: " << (samples > 0 ? (double) total_evals/samples : 0) << endl;

  // free mj_model and data, deactivate
  mj_deleteData(mj_data);
  mj_deleteModel(mj_model);
  mj_deactivate();
#ifdef VIS_EACH
  delete mj_vis;
#endif

  return pop;
}

// Step a generation
int pupper_generation(
    Population *pop, mjModel *mj_model, mjData *mj_data, MjVis *mj_vis,
    int generation, const char *filename, int &winner_num, int &winner_genes, int &winner_nodes) {
  vector<Organism *>::iterator cur_org;
  vector<Species *>::iterator cur_species;

  bool win = false;

  // Evaluate each organism on a test
  for (cur_org = (pop->organisms).begin(); cur_org!=(pop->organisms).end(); ++cur_org) {
    if (pupper_evaluate(*cur_org, mj_model, mj_data, mj_vis)) {
      win = true;
      winner_num = (*cur_org)->gnome->genome_id;
      winner_genes = (*cur_org)->gnome->extrons();
      winner_nodes = ((*cur_org)->gnome->nodes).size();
      if (winner_nodes==5) {
        (*cur_org)->gnome->print_to_filename("pupper_out/optimal");
      }
    }
  }

  // Average and max their fitnesses for dumping to file and snapshot
  for (cur_species = (pop->species).begin(); cur_species!=(pop->species).end(); ++cur_species) {

    // This experiment control routine issues commands to collect ave
    // and max fitness, as opposed to having the snapshot do it,
    // because this allows flexibility in terms of what time
    // to observe fitnesses at

    (*cur_species)->compute_average_fitness();
    (*cur_species)->compute_max_fitness();
  }

  // Take a snapshot of the population, so that it can be
  // visualized later on
  // if ((generation%1)==0) {
  //   pop->snapshot();
  // }


  // Only print to file every print_every generations
  if (win ||
      ((generation%(NEAT::print_every))==0)) {
    pop->print_to_file_by_species(filename);
  }

  if (win) {
    for (cur_org = (pop->organisms).begin(); cur_org!=(pop->organisms).end(); ++cur_org) {
      if ((*cur_org)->winner) {
        cout << "WINNER IS #" << ((*cur_org)->gnome)->genome_id << endl;
        print_Genome_tofile((*cur_org)->gnome, "pupper_out/winner");
      }
    }

  }

  pop->epoch(generation);

  if (win)
    return 1;
  else
    return 0;

}

tuple<bool, double> mujoco_evaluate(Network *net, mjModel *mj_model, mjData *mj_data, MjVis *mj_vis) {
  mj_resetData(mj_model, mj_data);

  float fitness = 0.0;

  int max_simulation_steps = 10000;

  // run the simulation
  bool failed = false;
  int cur_step = 0;
  while (!failed) {

    // collect sensor data and pass them to the network
    vector<double> sensors {};
    // position, dim 19
    for (int i = 0; i < mj_model->nq; i++) {
      sensors.push_back(mj_data->qpos[i]);
    }
    // velocity, dim 18
    for (int i = 0; i < mj_model->nv; i++) {
      sensors.push_back(mj_data->qvel[i]);
    }
    // actuator activation, dim 0
    // TODO @incomplete: why is this 0?
    //     if we don't know the activation position,
    //     then we don't know the current state of the joints,
    //     then how can we as human decide what to do next?
    for (int i = 0; i < mj_model->na; i++) {
      sensors.push_back(mj_data->act[i]);
    }
    net->load_sensors(sensors);

    // debug
    cout << "sensor data:"
      << " nq=" << mj_model->nq
      << " nv=" << mj_model->nv
      << " na=" << mj_model->na
      << endl;
    for (int i = 0; i < 7; i++) {
      cout << sensors[i] << "\t";
    }
    cout << endl;
    for (int i = 7; i < mj_model->nq; i++) {
      cout << sensors[i] << "\t";
      if ((i-7) > 0 && (i-7+1)%3 == 0) {
        cout << endl;
      }
    }
    for (int i = mj_model->nq; i < mj_model->nq + mj_model->nv; i++) {
      cout << sensors[i] << "\t";
      if ((i-mj_model->nq) > 0 && (i-mj_model->nq+1)%3 == 0) {
        cout << endl;
      }
    }

    // Activate the net
    // If it loops, exit returning only fitness of 0
    if (!(net->activate())) {
      return tuple(true, 0.0);
    }

    cout << "output size: " << net->outputs.size() << endl;
    cout << "control size: " << mj_model->nu << endl;

    for (int i = 0; i < net->outputs.size(); i++) {
      mj_data->ctrl[i] = net->outputs[i]->activation;
    }

    // TODO @incomplete: bounds check
    // if it fails, stop it now
    // if (outsideBounds()) {
    //   break;
    // }

    mj_step(mj_model, mj_data);

#ifdef VIS_EACH
    mj_vis->draw();
#endif
    
    cur_step += 1;


    double rot[4] = {mj_data->qpos[3], mj_data->qpos[4], mj_data->qpos[5], mj_data->qpos[6]};
    double res[3] = {0, 0, 0};
    double z[3] = {0, 0, 1};
    mju_rotVecQuat(res, z, rot);
    double cosine = mju_dot(res, z, 3) / mju_norm(res, 3);

    if (cosine < mju_sqrt(2) / 2) {
      cout << "failed rotation = " << rot[0] << " " << rot[1] << " " << rot[2] << " " << rot[3] << endl;
      cout << "failed direction = " << res[0] << " " << res[1] << " " << res[2] << endl;
      failed = true;
    }
  }

  // TODO @incomplete: add straightness, speed and orientation to fitness
  fitness = pow(mj_data->qpos[0], 2) + pow(mj_data->qpos[1], 2);
  fitness = sqrt(fitness);
  
  return tuple(true, fitness);
}

bool pupper_evaluate(Organism *org, mjModel *mj_model, mjData *mj_data, MjVis *mj_vis) {
  Network *net = org->net;

  // evaluate with mujoco
  auto eval_result = mujoco_evaluate(net, mj_model, mj_data, mj_vis);
  if (get<0>(eval_result)) {
    org->fitness = get<1>(eval_result);
  } else {
    exit(EXIT_FAILURE);
  }

#ifndef NO_SCREEN_OUT
  if (org->pop_champ_child) {
    cout << " <<DUPLICATE OF CHAMPION>> ";
  }

  // Output to screen
  cout << "Org " << (org->gnome)->genome_id << " fitness: " << org->fitness;
  cout << " (" << (org->gnome)->genes.size();
  cout << " / " << (org->gnome)->nodes.size() << ")";
  cout << "   ";
  if (org->mut_struct_baby)
    cout << " [struct]";
  if (org->mate_baby)
    cout << " [mate]";
  cout << endl;
#endif

  return false;
  // TODO @incomplete: what is a winner
  //
  // if ((!(thecart->generalization_test)) && (!(thecart->nmarkov_long)))
  //   if (org->pop_champ_child) {
  //     cout << org->gnome << endl;
  //     //DEBUG CHECK
  //     if (org->high_fit > org->fitness) {
  //       cout << "ALERT: ORGANISM DAMAGED" << endl;
  //       print_Genome_tofile(org->gnome, "failure_champ_genome");
  //       cin >> pause;
  //     }
  //   }
  //
  // //Decide if its a winner, in Markov Case
  // if (thecart->MARKOV) {
  //   if (org->fitness >= (thecart->maxFitness - 1)) {
  //     org->winner = true;
  //     return true;
  //   } else {
  //     org->winner = false;
  //     return false;
  //   }
  // }
  //   //if doing the long test non-markov
  // else if (thecart->nmarkov_long) {
  //   if (org->fitness >= 99999) {
  //     //if (org->fitness>=9000) {
  //     org->winner = true;
  //     return true;
  //   } else {
  //     org->winner = false;
  //     return false;
  //   }
  // } else if (thecart->generalization_test) {
  //   if (org->fitness >= 999) {
  //     org->winner = true;
  //     return true;
  //   } else {
  //     org->winner = false;
  //     return false;
  //   }
  // } else {
  //   org->winner = false;
  //   return false;  //Winners not decided here in non-Markov
  // }
}

MjVis::MjVis(mjModel *mj_model, mjData *mj_data) {

  this->mj_model = mj_model;
  this->mj_data = mj_data;

  if (!glfwInit()) {
    mju_error("Could not initialize GLFW");
  }

  // create window, make OpenGL context current, request v-sync
  window = glfwCreateWindow(1200, 900, "Demo", nullptr, nullptr);
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  // initialize visualization data structures
  mjv_defaultCamera(&cam);
  mjv_defaultOption(&opt);
  mjv_defaultScene(&scn);
  mjr_defaultContext(&con);

  // create scene and context
  mjv_makeScene(mj_model, &scn, 2000);
  mjr_makeContext(mj_model, &con, mjFONTSCALE_150);
}

MjVis::~MjVis() {
  //free visualization storage
  mjv_freeScene(&scn);
  mjr_freeContext(&con);
}

void MjVis::draw() {
  // get framebuffer viewport
  mjrRect viewport = {0, 0, 0, 0};
  glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

  // update scene and render
  mjv_updateScene(mj_model, mj_data, &opt, nullptr, &cam, mjCAT_ALL, &scn);
  mjr_render(viewport, &scn, &con);

  // swap OpenGL buffers (blocking call due to v-sync)
  glfwSwapBuffers(window);

  // process pending GUI events, call GLFW callbacks
  glfwPollEvents();
}