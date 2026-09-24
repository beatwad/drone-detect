// Cycle count of one FINN node for one frame, driven at full rate.
// Built against the node's pyverilator objects (see tb.sh):
//   -DTB_TOP=V<node> [-DHAS_WEIGHTS]   (the header V<node>.h is derived from it)
// Data is never driven: every node timed here is data-independent in timing.
// Input valid is held high until IN_WORDS words are accepted, output ready is
// held high, and weights (if the node has a weight stream) are always valid --
// i.e. an ideal weight streamer. Prints the cycles until OUT_WORDS outputs.
#include <cstdio>
#include <cstdlib>
#include "verilated.h"
#define TB_STR2(x) #x
#define TB_STR(x) TB_STR2(x)
#include TB_STR(TB_TOP.h)

double sc_time_stamp() { return 0; }

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: tb IN_WORDS OUT_WORDS [W_PERIOD]\n"); return 2; }
    long in_words = atol(argv[1]), out_words = atol(argv[2]);
    long w_period = argc > 3 ? atol(argv[3]) : 1;   // weights valid 1 cycle in W_PERIOD
    TB_TOP *t = new TB_TOP;
    t->ap_clk = 0; t->ap_rst_n = 0;
    for (int i = 0; i < 16; i++) { t->ap_clk = 1; t->eval(); t->ap_clk = 0; t->eval(); }
    t->ap_rst_n = 1;
    long sent = 0, got = 0, cyc = 0, first_out = -1;
    const long limit = 50 * (in_words + out_words) + 10000000;
    while (got < out_words && cyc < limit) {
        t->in0_V_TVALID = sent < in_words;
        t->out_V_TREADY = 1;
#ifdef HAS_WEIGHTS
        t->weights_V_TVALID = (cyc % w_period) == 0;
#endif
        t->eval();
        bool in_hs = t->in0_V_TVALID && t->in0_V_TREADY;
        bool out_hs = t->out_V_TVALID && t->out_V_TREADY;
        t->ap_clk = 1; t->eval(); t->ap_clk = 0; t->eval();
        if (in_hs) sent++;
        if (out_hs) { got++; if (first_out < 0) first_out = cyc; }
        cyc++;
    }
    printf("cycles %ld  in %ld/%ld  out %ld/%ld  first_out %ld%s\n", cyc, sent, in_words,
           got, out_words, first_out, cyc >= limit ? "  TIMEOUT" : "");
    delete t;
    return 0;
}
