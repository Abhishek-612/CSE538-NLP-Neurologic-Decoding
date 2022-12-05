for beam in 6;do
	for prune in 100 200 1000 2000 10000;do
		#for beta in 1 5 10 50;do
		for beta in 1;do
			./decode.sh recipes-until-50 recipe_GPT2/ recipes-until-50-$beam-$prune-$beta $beam $prune $beta
		done
	done
done
