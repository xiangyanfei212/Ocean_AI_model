
dir="/work/home/acrzcyisbk/ERA5_monthly"

# var='10m_u_component_of_wind' 
# var='10m_v_component_of_wind'
# var='2m_temperature'
# var='mean_sea_level_pressure'
# var='sea_surface_temperature'
var='surface_pressure'


in_dir="${dir}/${var}/degree_025"
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
