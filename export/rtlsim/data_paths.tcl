# Worst DATA paths of the shipping build, with the reset tree excluded from
# analysis only (nothing is re-placed or saved). Answers: what Fmax does the
# datapath itself allow, once the proc_sys_reset fan-out is set aside?
open_checkpoint /home/alex/finn_build_mdanilow/zynq_drone/finn_zynq_link.runs/impl_1/top_wrapper_routed.dcp
set rst [get_cells -hier -filter {NAME =~ top_i/rst_zynq_ps_99M/*}]
puts "reset cells excluded: [llength $rst]"
set_false_path -from $rst
report_timing -max_paths 15 -nworst 1 -sort_by slack -setup -file [pwd]/data_paths.rpt
