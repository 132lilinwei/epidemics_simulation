#include "crun.h"

static inline bool check_double_same(double *a, double *b, int length){
  int i = 0;
  for (i = 0; i < length; i++) {
    if ((a[i] != a[i] || b[i]!=b[i])) {
      outmsg("NAN found");
      return false;
    }
    if ((a[i] - b[i] > 1E-6) || (b[i] - a[i] > 1E-6)){
      outmsg("Not same at %d,  %.20lf, %.20lf", i, a[i], b[i]);
      return false;
    }
  }
  return true;
}


// float * compute_all_weights_cuda(state_t *s);
void compute_all_weights_cuda(state_t *s);
void find_all_sums_cuda(state_t *s);
void init_cuda(state_t *s);
void clean_cuda();


static inline void writeFile(state_t *s, FILE *outputFile) {
    int nid;
    graph_t *g = s->g;
    for(nid = 0; nid < g->nnode; nid++) {
        fprintf(outputFile, "%d %d %d\n", s->rat_count[nid], s->exposed_rat_count[nid], s->infectious_rat_count[nid]);
    }
}

static inline void write_num(state_t *s, FILE *file) {
    fprintf(file, "%d %d %d %d\n", s->rat_status_count[0], s->rat_status_count[1], s->rat_status_count[2], s->rat_status_count[3]);
}

/*  new version to compute node infectious probability
    It is based on the density of exposed and infectious
    The formula could be prob = alpha * density_of_infectious + beta * density_of_exposed
*/
static inline void compute_infection_prob(state_t *s) {
    START_ACTIVITY(ACTIVITY_PROB);
    int nnode = s->g->nnode;
    int nid;
    int *rat_count = s->rat_count;

    #pragma omp parallel for schedule(dynamic, 100)
    for(nid = 0; nid < nnode; nid++) {
        if(rat_count[nid] == 0) {
            s->node_prob[nid] = 0.0;
        } else {
            double density_of_exposed = 1.0 * s->exposed_rat_count[nid] / rat_count[nid];
            double density_of_infectious = 1.0 * s->infectious_rat_count[nid] / rat_count[nid];
            s->node_prob[nid] = s->alpha * density_of_infectious + s->beta * density_of_exposed;
            // printf("%lf\n", density_of_infectious);
        }
        // printf("%lf\n", s->node_prob[nid]);
    }
    FINISH_ACTIVITY(ACTIVITY_PROB);
}

/*  update each rat's status
    For status 0 (suspicious): random a number, if > node_prob, convert to 1 (exposed)
    For status 1 (exposed): use sigmoid 1 / (1 + e^{-(x - 7)}), x is the time of being exposed, convert to 2(infectious)
    For status 2 (exposed): use sigmoid 1 / (1 + e^{-(x - 7)}), x is the time of being infectious, convert to 3(recover)
    For status 3, won't be infected again.
 */
static inline void update_status(state_t *s, int bstart, int bcount) {
    START_ACTIVITY(ACTIVITY_UPDATE);
    double e = 2.718;    // set a constant here
    double *node_prob = s->node_prob;
    int *rat_position = s->rat_position;
    // int nrat = s->nrat;
    int rid;

    #pragma omp parallel for schedule(dynamic, 300)
    for(rid = bstart; rid < bstart + bcount; rid++) {
        int nid = rat_position[rid];
        random_t *seedp = &s->rat_seed[rid];
        double prob = next_random_float(seedp, 1);  // get random number
        // printf("%lf\n", node_prob[nid]);
        if(s->rat_status[rid] == 0){
            if(prob < node_prob[nid]) { // if 0 -> 1, change status, change node rat count
                #pragma omp atomic
                s->rat_status_count[0]--;
                #pragma omp atomic
                s->rat_status_count[1]++;
                #pragma omp atomic
                s->exposed_rat_count[nid]++;
                

                s->rat_status[rid] = 1;
            }
        } else if(s->rat_status[rid] == 1) {
            double exposed_to_infectious_prob = 1.0 / (1 + pow(e, -(s->rat_status_time[rid] - 7)));
            if(prob < exposed_to_infectious_prob) { // if 1 -> 2, change status, change node rat count, reset status time
                #pragma omp atomic
                s->rat_status_count[1]--;
                #pragma omp atomic
                s->rat_status_count[2]++;
                #pragma omp atomic
                s->exposed_rat_count[nid]--;
                #pragma omp atomic
                s->infectious_rat_count[nid]++;
                // printf("%lf\n", s->infectious_rat_count[nid]);

                s->rat_status[rid] = 2;
                s->rat_status_time[rid] = 0;
            } else {
                s->rat_status_time[rid]++;
            }
        } else if(s->rat_status[rid] == 2) {
            double infectious_to_recover_prob = 1.0 / (1 + pow(e, -(s->rat_status_time[rid] - 7)));
            if(prob < infectious_to_recover_prob) { // if 2 -> 3, change status, change node rat count, reset status time
                #pragma omp atomic
                s->rat_status_count[2]--;
                #pragma omp atomic
                s->rat_status_count[3]++;
                #pragma omp atomic
                s->infectious_rat_count[nid]--;
                s->rat_status[rid] = 3;
                s->rat_status_time[rid] = 0;
            } else {
                s->rat_status_time[rid]++;
            }
        }
    }
    FINISH_ACTIVITY(ACTIVITY_UPDATE);
}

/* Compute ideal load factor (ILF) for node */
static inline double neighbor_ilf(state_t *s, int nid) {
    graph_t *g = s->g;
    int outdegree = g->neighbor_start[nid+1] - g->neighbor_start[nid] - 1;
    int *start = &g->neighbor[g->neighbor_start[nid]+1];
    int i;
    double sum = 0.0;
    for (i = 0; i < outdegree; i++) {
       
        double ldensity = (s->rat_count[nid] == 0) ? 0.0 : 1.0 * s->infectious_rat_count[nid] / s->rat_count[nid];
        double rdensity = (s->rat_count[start[i]] == 0) ? 0.0 : 1.0 * s->infectious_rat_count[start[i]] / s->rat_count[start[i]];
        double r = (ldensity == 0.0 && rdensity == 0.0) ? 0.0 : imbalance_density(ldensity, rdensity);
        sum += r;
    }
    // change to a new ilf, where each node has different initial base ilf
    double ilf = BASE_ILF * (s->initial_load_factor[nid] / s->load_factor) + ILF_VARIABILITY * (sum / outdegree);
    return ilf;
}

/* Compute weight for node nid */
static inline double compute_weight(state_t *s, int nid) {
    int count = s->rat_count[nid];
    double ilf = neighbor_ilf(s, nid);
    return mweight((double) count/s->load_factor, ilf);
}

#if DEBUG
/** USEFUL DEBUGGING CODE **/
static void show_weights(state_t *s) {
    int nid, eid;
    graph_t *g = s->g;
    int nnode = g->nnode;
    int *neighbor = g->neighbor;
    outmsg("Weights\n");
    for (nid = 0; nid < nnode; nid++) {
	int eid_start = g->neighbor_start[nid];
	int eid_end  = g->neighbor_start[nid+1];
	outmsg("%d: [sum = %.3f]", nid, compute_sum_weight(s, nid));
	for (eid = eid_start; eid < eid_end; eid++) {
	    outmsg(" %.3f", compute_weight(s, neighbor[eid]));
	}
	outmsg("\n");
    }
}
#endif

/*
  Compute all initial node counts according to rat population.
  Assumes that rat position array is initially zeroed out.
 */
static inline void take_census(state_t *s) {
    int *rat_position = s->rat_position;
    int *rat_count = s->rat_count;
    int nrat = s->nrat;
    int ri;
    for (ri = 0; ri < nrat; ri++) {
	   rat_count[rat_position[ri]]++;
    }
}

/* Recompute all node weights */
static inline void compute_all_weights(state_t *s) {
    graph_t *g = s->g;
    double *node_weight = s->node_weight;

    START_ACTIVITY(ACTIVITY_WEIGHTS);
    int nid, hubid;
    #pragma omp parallel
    {
      #pragma omp for schedule(dynamic, 100) nowait
      for (hubid = 0; hubid < g->nhub; hubid++) {
        nid = g->hub[hubid];
        node_weight[nid] = compute_weight(s, nid);
      }
      #pragma omp for schedule(dynamic, 100) nowait
      for (nid = 0; nid < g->nnode; nid++) {
        if (g->mask[nid]) {
          node_weight[nid] = compute_weight(s, nid);
        }
      }
    }
    FINISH_ACTIVITY(ACTIVITY_WEIGHTS);
}

/* Precompute sums for each region */
static inline void find_all_sums(state_t *s) {
    graph_t *g = s->g;
    START_ACTIVITY(ACTIVITY_SUMS);

    #pragma omp parallel
    {
      int nid, hubid;
      #pragma omp for schedule(dynamic, 100) nowait
      for (hubid = 0; hubid < g->nhub; hubid++) {
        int eid;
        nid = g->hub[hubid];
        double sum = 0.0;
        for (eid = g->neighbor_start[nid]; eid < g->neighbor_start[nid+1]; eid++) { // this eid is just index of the neighbor in the neighbor array
            sum += s->node_weight[g->neighbor[eid]];
            s->neighbor_accum_weight[eid] = sum;
        }
        s->sum_weight[nid] = sum;
      }

      #pragma omp for schedule(dynamic, 300) nowait
      for (nid = 0; nid < g->nnode; nid++) {
        int eid;
        if (g->mask[nid]) {
            double sum = 0.0;
            for (eid = g->neighbor_start[nid]; eid < g->neighbor_start[nid+1]; eid++) { // this eid is just index of the neighbor in the neighbor array
                sum += s->node_weight[g->neighbor[eid]];
                s->neighbor_accum_weight[eid] = sum;
            }
            s->sum_weight[nid] = sum;
        }
      }
    }
    FINISH_ACTIVITY(ACTIVITY_SUMS);
}

// /* Recompute all node weights for float*/
// static inline void compute_all_weights_cuda_wrapper(state_t *s) {
//     START_ACTIVITY(ACTIVITY_WEIGHTS);
//     float * float_weights = compute_all_weights_cuda(s);
//     FINISH_ACTIVITY(ACTIVITY_WEIGHTS);
//     int i;
//     #pragma omp parallel for schedule(dynamic, 300)
//     for (i = 0; i < s->g->nnode; i++) {
//       s->node_weight[i] = (double) float_weights[i];
//     }
//     free(float_weights);

//     // double *temp = (double *) malloc(sizeof(double) * s->g->nnode);
//     // memcpy(temp, s->node_weight, sizeof(double) * s->g->nnode);
//     // compute_all_weights(s);
//     // check_double_same(temp, s->node_weight, s->g->nnode);
//     // free(temp);
// }

/* Recompute all node weights */
static inline void compute_all_weights_cuda_wrapper(state_t *s) {
    START_ACTIVITY(ACTIVITY_WEIGHTS);
    compute_all_weights_cuda(s);
    FINISH_ACTIVITY(ACTIVITY_WEIGHTS);

    // double *temp = (double *) malloc(sizeof(double) * s->g->nnode);
    // memcpy(temp, s->node_weight, sizeof(double) * s->g->nnode);
    // compute_all_weights(s);
    // check_double_same(temp, s->node_weight, s->g->nnode);
    // free(temp);
}

static inline void find_all_sums_cuda_wrapper(state_t *s) {
    // START_ACTIVITY(ACTIVITY_SUMS);
    // find_all_sums_cuda(s);
    // FINISH_ACTIVITY(ACTIVITY_SUMS);

    // double *temp_1 = (double *) malloc(sizeof(double) * s->g->nnode);
    // double *temp_2 = (double *) malloc(sizeof(double) * (s->g->nnode + s->g->nedge));
    // memcpy(temp_1, s->sum_weight, sizeof(double) * s->g->nnode);
    // memcpy(temp_2, s->neighbor_accum_weight, sizeof(double) * (s->g->nnode + s->g->nedge));
    find_all_sums(s);
    // check_double_same(temp_1, s->sum_weight, s->g->nnode);
    // check_double_same(temp_2, s->neighbor_accum_weight, (s->g->nnode + s->g->nedge));
    // free(temp_1);
    // free(temp_2);
 }


 static inline void compute_infection_prob_cuda_wrapper(state_t *s) {
    // START_ACTIVITY(ACTIVITY_PROB);
    // compute_infection_prob_cuda(s);
    // FINISH_ACTIVITY(ACTIVITY_PROB);

    // double *temp = (double *) malloc(sizeof(double) * s->g->nnode);
    // memcpy(temp, s->node_prob, sizeof(double) * s->g->nnode);
    compute_infection_prob(s);
    // check_double_same(temp, s->node_prob, s->g->nnode);
    // free(temp);
 }

/*
  Given list of increasing numbers, and target number,
  find index of first one where target is less than list value
*/

/*
  Linear search
 */
static inline int locate_value_linear(double target, double *list, int len) {
    int i;
    for (i = 0; i < len; i++)
	if (target < list[i])
	    return i;
    /* Shouldn't get here */
    return -1;
}
/*
  Binary search down to threshold, and then linear
 */
static inline int locate_value(double target, double *list, int len) {
    int left = 0;
    int right = len-1;
    while (left < right) {
	if (right-left+1 < BINARY_THRESHOLD)
	    return left + locate_value_linear(target, list+left, right-left+1);
	int mid = left + (right-left)/2;
	if (target < list[mid])
	    right = mid;
	else
	    left = mid+1;
    }
    return right;
}


/*
  This function assumes that node weights are already valid,
  and that have already computed sum of weights for each node,
  as well as cumulative weight for each neighbor
  Given list of integer counts, generate real-valued weights
  and use these to flip random coin returning value between 0 and len-1
*/
static inline int fast_next_random_move(state_t *s, int r) {
    int nid = s->rat_position[r];
    graph_t *g = s->g;
    random_t *seedp = &s->rat_seed[r];
    /* Guaranteed that have computed sum of weights */
    double tsum = s->sum_weight[nid];    
    double val = next_random_float(seedp, tsum);

    int estart = g->neighbor_start[nid];
    int elen = g->neighbor_start[nid+1] - estart;
    int offset = locate_value(val, &s->neighbor_accum_weight[estart], elen);
#if DEBUG
    if (offset < 0) {
	/* Shouldn't get here */
	outmsg("Internal error.  fast_next_random_move.  Didn't find valid move.  Target = %.2f/%.2f.\n",
	       val, tsum);
	return 0;
    }
#endif
    return g->neighbor[estart + offset];
}


/* Process single batch */
static inline void do_batch(state_t *s, int bstart, int bcount) {
    int ni, ri;
    graph_t *g = s->g;
    int nnode = g->nnode;
    int *rat_status = s->rat_status;
    /* Update weights */
    compute_all_weights_cuda_wrapper(s);

    find_all_sums_cuda_wrapper(s);


    START_ACTIVITY(ACTIVITY_NEXT);
    #pragma omp parallel 
    {
        int tid = omp_get_thread_num();
        int tcount = omp_get_num_threads();

        #pragma omp for schedule(dynamic, 100)
        for (ri = 0; ri < bcount; ri++) {
            int rid = ri+bstart;
            int onid = s->rat_position[rid];
            int nnid = fast_next_random_move(s, rid);
            s->rat_position[rid] = nnid;
            
            // update exposed and infectious rat count
            #pragma omp atomic
            s->rat_count[onid] -= 1;
            #pragma omp atomic
            s->rat_count[nnid] += 1;

            if(rat_status[rid] == 1) {
                #pragma omp atomic
                s->exposed_rat_count[onid] -= 1;
                #pragma omp atomic
                s->exposed_rat_count[nnid] += 1;
            } else if(rat_status[rid] == 2) {
                // printf("%d\n", s->infectious_rat_count[onid]);
                // printf("%d\n", s->infectious_rat_count[nnid]);
                #pragma omp atomic
                s->infectious_rat_count[onid] -= 1;
                #pragma omp atomic
                s->infectious_rat_count[nnid] += 1;
            }
        }
    }
    FINISH_ACTIVITY(ACTIVITY_NEXT);

    // only update in each step rather than one batch (???)
    compute_infection_prob_cuda_wrapper(s);
    update_status(s, bstart, bcount);


}

static void batch_step(state_t *s) {
    int rid = 0;
    int bsize = s->batch_size;
    int nrat = s->nrat;
    int bcount;
    while (rid < nrat) {
	bcount = nrat - rid;
	if (bcount > bsize)
	    bcount = bsize;
	do_batch(s, rid, bcount);
	rid += bcount;
  
    }
}

// get the hub node, for each node, compute initial load factor
static inline void divide_graph(state_t *s) {
  graph_t *g = s->g;
  int nnode = g->nnode;
  int nid;
  int hubid = 0;
  int *rat_count = s->rat_count;
  for (nid = 0; nid < nnode; nid++) {
    int outdegree = g->neighbor_start[nid+1] - g->neighbor_start[nid] - 1;
    if (outdegree > HUB_THREASHOLD) { // the threshold does not matter, just need to be more than 4
      g->hub[hubid++] = nid;
      g->mask[nid] = false;
    } else {
      g->mask[nid] = true;
    }

    // compute initial load factor
    s->initial_load_factor[nid] = 1.0 * rat_count[nid] / s->load_factor;
  }
  g->nhub = hubid;
}


double simulate(state_t *s, int count, update_t update_mode, int dinterval, bool display, char* output) {
    int i;
    /* Adjust bath size if not in bath mode */
    if (update_mode == UPDATE_SYNCHRONOUS) 
      s->batch_size = s->nrat;
    else if (update_mode == UPDATE_RAT) 
      s->batch_size = 1;


    FILE *file = NULL;
    FILE *outputFile = NULL;
    /* Compute and show initial state */
    if(output != NULL) {
        char filepath[100];
        char outputfilepath [100];
        strcpy(filepath, output);
        strcpy(outputfilepath, output);
        strcat(filepath, "_status_count.txt");
        strcat(outputfilepath, "_node_count.txt");
        file = fopen(filepath, "w+");
        outputFile = fopen(outputfilepath, "w+");
    }


    bool show_counts = true;
    double start = currentSeconds();
    take_census(s);
    divide_graph(s);
    init_cuda(s);

    if (display) show(s, show_counts);

    for (i = 0; i < count; i++) {
  	  batch_step(s);
      if(output != NULL) {
          write_num(s, file);
          writeFile(s, outputFile);
      }
    	if (display) {
    	    show_counts = (((i+1) % dinterval) == 0) || (i == count-1);
    	    show(s, show_counts);
    	}
    }
    double delta = currentSeconds() - start;
    
    if(outputFile != NULL) {
        fclose(file);
        fclose(outputFile);
    }

    clean_cuda();
    done();
    return delta;
}
