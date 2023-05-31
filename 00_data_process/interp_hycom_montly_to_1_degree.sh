conda activate cdo

var='ts3z'
# var='uv3z'
# var='ssh'

dir="/work/home/acrzcyisbk/HYCOM_monthly"
in_dir="${dir}/${var}/degree_0083"
out_dir="${dir}/${var}/degree_1"

mkdir -p ${out_dir}

for i in `find ${in_dir} -name "*.nc" -maxdepth 1`
do
    filename=`basename $i`
    outfile="${out_dir}/${filename}"

    if [ ! -f ${outfile} ]
    then
        echo "------------------------------------------------"
        echo "Interplating $i"

        # Horizontal interpolation according to targetgrid.txt(EN4_ana)
        cdo -P 8 remapbil,targetgrid.txt ${i} ${outfile}

    else
        echo "${outfile} exists, skip"
    fi

done
