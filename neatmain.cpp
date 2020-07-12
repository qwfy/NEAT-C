/*
 Copyright 2001 The University of Texas at Austin

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#include <iostream>
#include <vector>
#include "neat.h"
#include "population.h"
#include "experiments.h"
#include "pupper_exp.h"
using namespace std;

int main(int argc, char *argv[]) {

  NEAT::Population *p = nullptr;

  // Seed the random-number generator with current time,
  // so that the numbers will be different every time we run
  srand((unsigned) time(nullptr));

  if (argc!=2) {
    cerr << "A NEAT parameters file (.ne file) is required to run the experiments" << endl;
    return -1;
  }

  // Load in the params
  bool isLoadSuccess = NEAT::load_neat_params(argv[1], true);
  if (!isLoadSuccess) {
    cerr << "Failed to load parameter file " << argv[1] << endl;
    return -1;
  }

  // cout << "Please choose an experiment:" << endl;
  // cout << "1 - 1-pole balancing" << endl;
  // cout << "2 - 2-pole balancing, velocity info provided" << endl;
  // cout << "3 - 2-pole balancing, no velocity info provided (non-markov)" << endl;
  // cout << "4 - XOR" << endl;
  // cout << "5 - Pupper" << endl;
  // cout << "Number: ";
  //
  // int choice;
  // cin >> choice;

  int choice = 5;

  switch (choice) {
  case 1:
    p = pole1_test(100);
    break;
  case 2:
    p = pole2_test(100, 1);
    break;
  case 3:
    p = pole2_test(100, 0);
    break;
  case 4:
    p = xor_test(100);
    break;
  case 5:
    p = pupper_simulate(100);
    break;
  default:
    cerr << "Bad choice" << endl;
    return -1;
  }

  if (p == nullptr) {
    return -1;
  } else {
    delete p;
    return 0;
  }
}

