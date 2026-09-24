// One MVAU_rtl layer as it is built in hardware: FINN's memstream IP streaming
// the weights out of on-chip memory into the MVU core, exactly as
// MatrixVectorActivation.code_generation_ipi() wires them for internal_decoupled.
// Parameters are set per layer from the command line (see layer.sh).
module layer_top #(
	parameter DEPTH = 288,
	parameter WIDTH = 32,
	parameter INIT_FILE = "",
	parameter IN_W = 32,
	parameter OUT_W = 16
)(
	input  ap_clk,
	input  ap_rst_n,
	input  [IN_W-1:0]  in0_V_TDATA,
	input              in0_V_TVALID,
	output             in0_V_TREADY,
	output [OUT_W-1:0] out_V_TDATA,
	output             out_V_TVALID,
	input              out_V_TREADY
);
	wire [WIDTH-1:0] w_tdata;
	wire             w_tvalid, w_tready;

	memstream_axi_wrapper #(.DEPTH(DEPTH), .WIDTH(WIDTH), .INIT_FILE(INIT_FILE), .RAM_STYLE("auto")) wstrm (
		.ap_clk(ap_clk), .ap_rst_n(ap_rst_n),
		.awvalid(1'b0), .awprot(3'b0), .awaddr('0), .wvalid(1'b0), .wdata('0), .wstrb('0), .bready(1'b0),
		.arvalid(1'b0), .arprot(3'b0), .araddr('0), .rready(1'b0),
		.awready(), .wready(), .bvalid(), .bresp(), .arready(), .rvalid(), .rresp(), .rdata(),
		.m_axis_0_tready(w_tready), .m_axis_0_tvalid(w_tvalid), .m_axis_0_tdata(w_tdata)
	);

	`LAYER mvu (
		.ap_clk(ap_clk), .ap_rst_n(ap_rst_n),
		.weights_V_TDATA(w_tdata), .weights_V_TVALID(w_tvalid), .weights_V_TREADY(w_tready),
		.in0_V_TDATA(in0_V_TDATA), .in0_V_TVALID(in0_V_TVALID), .in0_V_TREADY(in0_V_TREADY),
		.out_V_TDATA(out_V_TDATA), .out_V_TVALID(out_V_TVALID), .out_V_TREADY(out_V_TREADY)
	);
endmodule
