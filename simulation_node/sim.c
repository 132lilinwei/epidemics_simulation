#include "crun.h"

static inline bool check_double_same(double *a, double *b, int length){
  int i = 0;
  for (i = 0; i < length; i++) {
    if ((a[i] != a[i] || b[i]!=b[i])) {
      outmsg("NAN found");
      return false;
    }
    if ((a[i] - b[i] > 1E-10) || (b[i] - a[i] > 1E-10)){
      outmsg("Not same at %d,  %.20lf, %.20lf", i, a[i], b[i]);
      return false;
    }
  }
  return true;
}


void init_cuda(state_t *s);
void clean_cuda();
void compute_all_weights_cuda(state_t *s);
void find_all_sums_cuda(state_t *s);
void next_move_cuda(state_t *s, double batch_fraction);
void compute_infection_prob_and_update_status_cuda(state_t *s, double batch_fraction);


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



/* Process single batch */
static inline void do_batch(state_t *s, double batch_fraction) {
    START_ACTIVITY(ACTIVITY_WEIGHTS);
    compute_all_weights_cuda(s);
    FINISH_ACTIVITY(ACTIVITY_WEIGHTS);

    START_ACTIVITY(ACTIVITY_SUMS);
    find_all_sums_cuda(s);
    FINISH_ACTIVITY(ACTIVITY_SUMS);

    START_ACTIVITY(ACTIVITY_NEXT);
    next_move_cuda(s, batch_fraction);
    FINISH_ACTIVITY(ACTIVITY_NEXT);

    START_ACTIVITY(ACTIVITY_UPDATE);
    compute_infection_prob_and_update_status_cuda(s, batch_fraction);
    FINISH_ACTIVITY(ACTIVITY_UPDATE);


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
	do_batch(s, (double) bcount / (double) nrat);
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
