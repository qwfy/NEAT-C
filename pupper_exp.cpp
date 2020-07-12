#include "experiments.h"
#include "pupper_exp.h"
#include <cstring>
#include <string>

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
  int exp_count;
  int samples;

  memset(evals, 0, NEAT::num_runs*sizeof(int));
  memset(genes, 0, NEAT::num_runs*sizeof(int));
  memset(nodes, 0, NEAT::num_runs*sizeof(int));

  ifstream input_file("pupper_start_genes.txt", ios::in);
  if (!input_file.is_open()) {
    cerr << "Failed to open start genes" << endl;
    return nullptr;
  }

  cout << "START PUPPER TEST" << endl;

  cout << "Reading in the start genome" << endl;
  input_file >> _param_name;
  input_file >> id;
  cout << "Reading in Genome id " << id << endl;
  start_genome = new Genome(id, input_file);
  input_file.close();

  for (exp_count = 0; exp_count < NEAT::num_runs; exp_count++) {
    // Spawn the Population
    cout << "Spawning Population off Genome2" << endl;

    pop = new Population(start_genome, NEAT::pop_size);

    cout << "Verifying Spawned Pop" << endl;
    pop->verify();

    for (gen = 1; gen <= gens; gen++) {
      cout << "Epoch " << gen << endl;

      // This is how to make a custom filename
      file_name_buf = new ostringstream();
      (*file_name_buf) << "pupper_out/gen_" << gen << ends;  //needs end marker

#ifndef NO_SCREEN_OUT
      cout << "name of fname: " << file_name_buf->str() << endl;
#endif

      string filename = file_name_buf->str();

      //Check for success
      if (pupper_epoch(pop, gen, filename.c_str(), winner_num, winner_genes, winner_nodes)) {
        //	if (pupper_epoch(pop,gen,file_name_buf->str(),winner_num,winner_genes,winner_nodes)) {
        //Collect Stats on end of experiment
        evals[exp_count] = NEAT::pop_size*(gen - 1) + winner_num;
        genes[exp_count] = winner_genes;
        nodes[exp_count] = winner_nodes;
        gen = gens;
      }

      //Clear output filename
      file_name_buf->clear();
      delete file_name_buf;
    }

    if (exp_count < NEAT::num_runs - 1) {
      delete pop;
    }
  }

  // Average and print stats
  cout << "Nodes: " << endl;
  for (exp_count = 0; exp_count < NEAT::num_runs; exp_count++) {
    cout << nodes[exp_count] << endl;
    total_nodes += nodes[exp_count];
  }

  cout << "Genes: " << endl;
  for (exp_count = 0; exp_count < NEAT::num_runs; exp_count++) {
    cout << genes[exp_count] << endl;
    total_genes += genes[exp_count];
  }

  cout << "Evals " << endl;
  samples = 0;
  for (exp_count = 0; exp_count < NEAT::num_runs; exp_count++) {
    cout << evals[exp_count] << endl;
    if (evals[exp_count] > 0) {
      total_evals += evals[exp_count];
      samples++;
    }
  }

  cout << "Failures: " << (NEAT::num_runs - samples) << " out of " << NEAT::num_runs << " runs" << endl;
  cout << "Average Nodes: " << (samples > 0 ? (double) total_nodes/samples : 0) << endl;
  cout << "Average Genes: " << (samples > 0 ? (double) total_genes/samples : 0) << endl;
  cout << "Average Evals: " << (samples > 0 ? (double) total_evals/samples : 0) << endl;

  return pop;

}

int pupper_epoch(Population *pop, int generation, const char *filename, int &winner_num, int &winner_genes, int &winner_nodes) {
  vector<Organism *>::iterator cur_org;
  vector<Species *>::iterator cur_species;

  bool win = false;

  // Evaluate each organism on a test
  for (cur_org = (pop->organisms).begin(); cur_org!=(pop->organisms).end(); ++cur_org) {
    if (pupper_evaluate(*cur_org)) {
      win = true;
      winner_num = (*cur_org)->gnome->genome_id;
      winner_genes = (*cur_org)->gnome->extrons();
      winner_nodes = ((*cur_org)->gnome->nodes).size();
      if (winner_nodes==5) {
        // You could dump out optimal genomes here if desired
        // (*cur_org)->gnome->print_to_filename("pupper_optimal");
        // cout<<"DUMPED OPTIMAL"<<endl;
      }
    }
  }

  // Average and max their fitnesses for dumping to file and snapshot
  for (cur_species = (pop->species).begin(); cur_species!=(pop->species).end(); ++cur_species) {

    //This experiment control routine issues commands to collect ave
    //and max fitness, as opposed to having the snapshot do it,
    //because this allows flexibility in terms of what time
    //to observe fitnesses at

    (*cur_species)->compute_average_fitness();
    (*cur_species)->compute_max_fitness();
  }

  // Take a snapshot of the population, so that it can be
  //visualized later on
  //if ((generation%1)==0)
  //  pop->snapshot();

  // Only print to file every print_every generations
  if (win ||
      ((generation%(NEAT::print_every))==0))
    pop->print_to_file_by_species(filename);

  if (win) {
    for (cur_org = (pop->organisms).begin(); cur_org!=(pop->organisms).end(); ++cur_org) {
      if ((*cur_org)->winner) {
        cout << "WINNER IS #" << ((*cur_org)->gnome)->genome_id << endl;
        //Prints the winner to file
        //IMPORTANT: This causes generational file output!
        print_Genome_tofile((*cur_org)->gnome, "pupper_winner");
      }
    }

  }

  pop->epoch(generation);

  if (win)
    return 1;
  else
    return 0;

}

bool pupper_evaluate(Organism *org) {
  Network *net;
  double out[4]; //The four outputs
  double this_out; //The current output
  int count;
  double error_sum;

  bool success;  //Check for successful activation
  int num_nodes;  /* Used to figure out how many nodes
		    should be visited during activation */

  int net_depth; //The max depth of the network to be activated
  int relax; //Activates until relaxation

  //The four possible input combinations to xor
  //The first number is for biasing
  double in[4][3] = {{1.0, 0.0, 0.0},
                     {1.0, 0.0, 1.0},
                     {1.0, 1.0, 0.0},
                     {1.0, 1.0, 1.0}};

  net = org->net;
  num_nodes = ((org->gnome)->nodes).size();

  net_depth = net->max_depth();

  //TEST CODE: REMOVE
  //cout<<"ACTIVATING: "<<org->gnome<<endl;
  //cout<<"DEPTH: "<<net_depth<<endl;

  //Load and activate the network on each input
  for (count = 0; count <= 3; count++) {
    net->load_sensors(in[count]);

    //Relax net and get output
    success = net->activate();

    //use depth to ensure relaxation
    for (relax = 0; relax <= net_depth; relax++) {
      success = net->activate();
      this_out = (*(net->outputs.begin()))->activation;
    }

    out[count] = (*(net->outputs.begin()))->activation;

    net->flush();

  }

  if (success) {
    error_sum = (fabs(out[0]) + fabs(1.0 - out[1]) + fabs(1.0 - out[2]) + fabs(out[3]));
    org->fitness = pow((4.0 - error_sum), 2);
    org->error = error_sum;
  } else {
    //The network is flawed (shouldnt happen)
    error_sum = 999.0;
    org->fitness = 0.001;
  }

#ifndef NO_SCREEN_OUT
  cout << "Org " << (org->gnome)->genome_id << "                                     error: " << error_sum << "  ["
       << out[0] << " " << out[1] << " " << out[2] << " " << out[3] << "]" << endl;
  cout << "Org " << (org->gnome)->genome_id << "                                     fitness: " << org->fitness << endl;
#endif

  //  if (error_sum<0.05) {
  //if (error_sum<0.2) {
  if ((out[0] < 0.5) && (out[1] >= 0.5) && (out[2] >= 0.5) && (out[3] < 0.5)) {
    org->winner = true;
    return true;
  } else {
    org->winner = false;
    return false;
  }

}