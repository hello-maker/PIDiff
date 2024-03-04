#for id in range(100):
#    !python scripts/sample_diffusion.py configs/sampling.yml --data_id {id} 

for id in {0..99}
do 
    python scripts/sample_diffusion.py configs/sampling.yml --data_id $id
done     

                                                                                                                                         
