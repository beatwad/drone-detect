// Behavioural stand-in for Xilinx axis_data_fifo v2.0, so the stitched design
// can be simulated in Verilator (the real IP wraps XPM macros).
//
// For throughput this is equivalent: a word accepted in one cycle is readable
// in the next. The real IP adds a couple of cycles of read latency, which is
// noise against ~10^6 cycles per frame.
//
// Capacity is C_FIFO_DEPTH * `FIFO_ALLOC words. The effective limit starts at
// C_FIFO_DEPTH -- i.e. exactly the built hardware -- and can be changed at run
// time without recompiling:
//   +fifo_scale=K          every instance to K x its built depth
//   +fifo:<%m>=D           one instance (hierarchical name) to D words
//   +fifo_list             print every instance with its depth at start
//   +fifo_report           print every instance's max occupancy at the end
//   +stat_from=C           also count, from cycle C on, the cycles this FIFO
//                          was full (producer blocked) and empty (consumer
//                          starved); printed with +fifo_report
`ifndef FIFO_ALLOC
`define FIFO_ALLOC 4
`endif
module axis_data_fifo_v2_0_9_top #(
	parameter C_FAMILY = "zynquplus",
	parameter integer C_AXIS_TDATA_WIDTH = 8,
	parameter integer C_AXIS_TID_WIDTH = 1,
	parameter integer C_AXIS_TDEST_WIDTH = 1,
	parameter integer C_AXIS_TUSER_WIDTH = 1,
	parameter [31:0]  C_AXIS_SIGNAL_SET = 32'h3,
	parameter integer C_FIFO_DEPTH = 16,
	parameter integer C_FIFO_MODE = 1,
	parameter integer C_IS_ACLK_ASYNC = 0,
	parameter integer C_SYNCHRONIZER_STAGE = 3,
	parameter integer C_ACLKEN_CONV_MODE = 0,
	parameter integer C_ECC_MODE = 0,
	parameter         C_FIFO_MEMORY_TYPE = "auto",
	parameter integer C_USE_ADV_FEATURES = 0,
	parameter integer C_PROG_EMPTY_THRESH = 5,
	parameter integer C_PROG_FULL_THRESH = 11
)(
	input  s_axis_aresetn, s_axis_aclk, s_axis_aclken,
	input  s_axis_tvalid,
	output s_axis_tready,
	input  [C_AXIS_TDATA_WIDTH-1:0] s_axis_tdata,
	input  [(C_AXIS_TDATA_WIDTH+7)/8-1:0] s_axis_tstrb, s_axis_tkeep,
	input  s_axis_tlast,
	input  [C_AXIS_TID_WIDTH-1:0] s_axis_tid,
	input  [C_AXIS_TDEST_WIDTH-1:0] s_axis_tdest,
	input  [C_AXIS_TUSER_WIDTH-1:0] s_axis_tuser,
	input  m_axis_aclk, m_axis_aclken,
	output m_axis_tvalid,
	input  m_axis_tready,
	output [C_AXIS_TDATA_WIDTH-1:0] m_axis_tdata,
	output [(C_AXIS_TDATA_WIDTH+7)/8-1:0] m_axis_tstrb, m_axis_tkeep,
	output m_axis_tlast,
	output [C_AXIS_TID_WIDTH-1:0] m_axis_tid,
	output [C_AXIS_TDEST_WIDTH-1:0] m_axis_tdest,
	output [C_AXIS_TUSER_WIDTH-1:0] m_axis_tuser,
	output [31:0] axis_wr_data_count, axis_rd_data_count,
	output almost_empty, prog_empty, almost_full, prog_full,
	output sbiterr, dbiterr,
	input  injectsbiterr, injectdbiterr
);
	localparam integer CAP = C_FIFO_DEPTH * `FIFO_ALLOC;
	localparam integer AW  = $clog2(CAP);

	logic [C_AXIS_TDATA_WIDTH-1:0] mem [0:CAP-1];
	logic [AW-1:0] wp, rp;
	int cnt, lim, maxc;
	longint cyc, from, n_full, n_empty, n_win;

	initial begin
		string key;
		int v;
		lim = C_FIFO_DEPTH;
		if ($value$plusargs("fifo_scale=%d", v)) lim = C_FIFO_DEPTH * v;
		key = $sformatf("fifo:%m=%%d");
		if ($value$plusargs(key, v)) lim = v;
		from = 64'h7fffffffffffffff;
		if ($value$plusargs("stat_from=%d", v)) from = v;
		cyc = 0; n_full = 0; n_empty = 0; n_win = 0;
		if (lim > CAP) lim = CAP;
		if ($test$plusargs("fifo_list"))
			$display("FIFO %m depth %0d width %0d limit %0d", C_FIFO_DEPTH, C_AXIS_TDATA_WIDTH, lim);
	end

	wire wr = s_axis_tvalid && s_axis_tready;
	wire rd = m_axis_tvalid && m_axis_tready;
	assign s_axis_tready = s_axis_aresetn && (cnt < lim);
	assign m_axis_tvalid = (cnt > 0);
	assign m_axis_tdata  = mem[rp];

	always @(posedge s_axis_aclk) begin
		if (!s_axis_aresetn) begin
			wp <= '0; rp <= '0; cnt <= 0; maxc <= 0;
		end else begin
			if (wr) begin
				mem[wp] <= s_axis_tdata;
				wp <= (wp == AW'(CAP-1)) ? '0 : wp + 1'b1;
			end
			if (rd) rp <= (rp == AW'(CAP-1)) ? '0 : rp + 1'b1;
			cnt <= cnt + (wr ? 1 : 0) - (rd ? 1 : 0);
			if (cnt > maxc) maxc <= cnt;
			cyc <= cyc + 1;
			if (cyc >= from) begin
				n_win <= n_win + 1;
				if (cnt >= lim) n_full <= n_full + 1;
				if (cnt == 0) n_empty <= n_empty + 1;
			end
		end
	end

	final if ($test$plusargs("fifo_report"))
		$display("FIFOMAX %m depth %0d limit %0d max %0d%s full %0.1f empty %0.1f", C_FIFO_DEPTH, lim, maxc,
		         maxc >= lim ? "  FULL" : "",
		         n_win ? 100.0 * n_full / n_win : 0.0, n_win ? 100.0 * n_empty / n_win : 0.0);

	assign m_axis_tstrb = '1; assign m_axis_tkeep = '1; assign m_axis_tlast = 1'b1;
	assign m_axis_tid = '0; assign m_axis_tdest = '0; assign m_axis_tuser = '0;
	assign axis_wr_data_count = 32'(cnt); assign axis_rd_data_count = 32'(cnt);
	assign almost_empty = (cnt <= 1); assign prog_empty = (cnt <= C_PROG_EMPTY_THRESH);
	assign almost_full = (cnt >= lim - 1); assign prog_full = (cnt >= C_PROG_FULL_THRESH);
	assign sbiterr = 1'b0; assign dbiterr = 1'b0;
endmodule
