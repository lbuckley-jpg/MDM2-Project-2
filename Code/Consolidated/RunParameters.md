note if any code doesnt run its due to a relabelling of cpto to bpto to be congruent with the slides. 

Here in the arguments cpto refers to bpto linear damping that is proportional to velocity.

python Analysis.py --tspan 60 --seed 42 --runs 3 --peakperiods 5.2 --significantwaveheights 2.5 --buoymasses 261800 --cptos 80000 --ptoforcemax 100000 --save



python CaptureWidthPlot.py --tspan 60 --seed 42 --peakperiods 5.2 6.01 7.05 8.0 10.0 12.0 15.0 --significantwaveheights 1.5 --cptos 36600 --buoymasses 100000 --buoyradius 2.0  --no-pontryagin

Additional cptos can be ran by adding --cptos <value> to the command.

also remove the --no-pontryagin flag to run pontryagin analysis.

