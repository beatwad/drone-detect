// Whole-accelerator timing: stream NF frames in at full rate, output always
// ready, print the cycle each output frame completes. Reproduces the board's
// latency L (first frame) and steady-state interval I (frame to frame).
//   design_tb [NF] [PERIOD] [+fifo_scale=K] [+fifo:<name>=D] [+fifo_list] [+fifo_report]
// PERIOD > 0 paces the input: frame f may not start before cycle f*PERIOD. With
// large FIFOs and a period just above the ideal interval, max occupancies are
// the steady-state need rather than a backlog from flooding the input.
#include <cstdio>
#include <cstdlib>
#include "verilated.h"
#include "VStreamingDataflowPartition_1_wrapper.h"

static const long IN_WORDS = 192 * 320 * 3;   // UINT8, one word per channel
static const long OUT_WORDS = 24 * 40 * 65;   // INT21 in 24-bit words

int main(int argc, char **argv) {
    auto *ctx = new VerilatedContext;
    ctx->commandArgs(argc, argv);
    long nf = (argc > 1 && argv[1][0] != '+') ? atol(argv[1]) : 4;
    long period = (argc > 2 && argv[2][0] != '+') ? atol(argv[2]) : 0;
    auto *t = new VStreamingDataflowPartition_1_wrapper{ctx};
    t->ap_clk = 0; t->ap_rst_n = 0;
    for (int i = 0; i < 32; i++) { t->ap_clk = 1; t->eval(); t->ap_clk = 0; t->eval(); }
    t->ap_rst_n = 1;
    long sent = 0, got = 0, cyc = 0, prev = 0;
    const long limit = nf * 20000000L;
    while (got < nf * OUT_WORDS && cyc < limit) {
        t->s_axis_0_tvalid = sent < nf * IN_WORDS && (period == 0 || cyc >= (sent / IN_WORDS) * period);
        t->m_axis_0_tready = 1;
        t->eval();
        bool in_hs = t->s_axis_0_tvalid && t->s_axis_0_tready;
        bool out_hs = t->m_axis_0_tvalid && t->m_axis_0_tready;
        t->ap_clk = 1; t->eval(); t->ap_clk = 0; t->eval();
        if (in_hs) sent++;
        cyc++;
        if (out_hs && ++got % OUT_WORDS == 0) {
            long f = got / OUT_WORDS;
            printf("frame %ld done at %10ld cycles  %s %10ld  (%.2f ms at 100 MHz)\n", f, cyc,
                   f == 1 ? "latency " : "interval", cyc - prev, (cyc - prev) * 1e-5);
            fflush(stdout);
            prev = cyc;
        }
        if (cyc % 1000000 == 0) { fprintf(stderr, "  .. %ld Mcycles, in %ld, out %ld\n", cyc / 1000000, sent, got); }
    }
    if (cyc >= limit) printf("TIMEOUT at %ld cycles, in %ld out %ld\n", cyc, sent, got);
    t->final();
    delete t;
    delete ctx;
    return 0;
}
