# example
# transfer /var/data/StackData/futures/team/huxo/factors_by_instru/MTM/*.ss file to
# /var/TSDB/FutHot/team/huxo/factors_by_instru/MTM

src_path=/var/data/StackData/futures
dst_tsdb=/var/TSDB/FutHot
tbl=d01e
prefix="team.huxo.factors_by_instru.MTM"
tp_end='20240322 17:00:00.000'
key=types
ncpu=40

qalign update -v --path $src_path --db $dst_tsdb --tbl $tbl --fields MTM001 --end "$tp_end" --ignore-missing-ticker --prefix $prefix --key $key  --ncpu $ncpu --exact --float32 --skip-ii-chk
qalign update -v --path $src_path --db $dst_tsdb --tbl $tbl --fields MTM003 --end "$tp_end" --ignore-missing-ticker --prefix $prefix --key $key  --ncpu $ncpu --exact --float32 --skip-ii-chk
qalign update -v --path $src_path --db $dst_tsdb --tbl $tbl --fields MTM005 --end "$tp_end" --ignore-missing-ticker --prefix $prefix --key $key  --ncpu $ncpu --exact --float32 --skip-ii-chk
qalign update -v --path $src_path --db $dst_tsdb --tbl $tbl --fields MTM010 --end "$tp_end" --ignore-missing-ticker --prefix $prefix --key $key  --ncpu $ncpu --exact --float32 --skip-ii-chk
qalign update -v --path $src_path --db $dst_tsdb --tbl $tbl --fields MTM020 --end "$tp_end" --ignore-missing-ticker --prefix $prefix --key $key  --ncpu $ncpu --exact --float32 --skip-ii-chk
qalign update -v --path $src_path --db $dst_tsdb --tbl $tbl --fields MTM060 --end "$tp_end" --ignore-missing-ticker --prefix $prefix --key $key  --ncpu $ncpu --exact --float32 --skip-ii-chk
qalign update -v --path $src_path --db $dst_tsdb --tbl $tbl --fields MTM120 --end "$tp_end" --ignore-missing-ticker --prefix $prefix --key $key  --ncpu $ncpu --exact --float32 --skip-ii-chk
qalign update -v --path $src_path --db $dst_tsdb --tbl $tbl --fields MTM180 --end "$tp_end" --ignore-missing-ticker --prefix $prefix --key $key  --ncpu $ncpu --exact --float32 --skip-ii-chk
qalign update -v --path $src_path --db $dst_tsdb --tbl $tbl --fields MTM240 --end "$tp_end" --ignore-missing-ticker --prefix $prefix --key $key  --ncpu $ncpu --exact --float32 --skip-ii-chk
