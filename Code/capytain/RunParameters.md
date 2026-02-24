## File to store run command formate to save time and track parameters

### !!! Do not Delete !!!

### For PointAbsorberSimulation.py

*Example runs (5 m radius buoy, mass 500 kg, deep-ish water; units: m, s, kg). Wave frequency is in Hz.*

#### Calm sea, long waves (small motions, mostly for sanity checks)

python PointAbsorberSimulation.py --buoymass 500 --buoyradius 5 --waterdensity 1000 --waterdepth 100 --wavefrequency 0.08 --wavedirection 3.14159 --waveamplitude 1.0

#### Moderate sea, near-resonant heave (strong response, no PTO)

python PointAbsorberSimulation.py --buoymass 500 --buoyradius 5 --waterdensity 1000 --waterdepth 100 --wavefrequency 0.20 --wavedirection 3.14159 --waveamplitude 2.0

#### Higher frequency sea (shorter waves, smaller heave response)

python PointAbsorberSimulation.py --buoymass 500 --buoyradius 5 --waterdensity 1000 --waterdepth 100 --wavefrequency 0.30 --wavedirection 3.14159 --waveamplitude 2.0


### For PTOPointAborberSimulation.py

*Same sea states as above, but with PTO in heave only. PTO values are order-of-magnitude comparable to hydrostatic stiffness and radiation damping for a 5 m sphere; adjust as needed when doing optimization.*

#### Near-resonant sea, weak PTO (close to free motion, good baseline)

python PTOPointAbsorberSimulation.py --buoymass 500 --buoyradius 5 --waterdensity 1000 --waterdepth 100 --wavefrequency 0.20 --wavedirection 3.14159 --waveamplitude 2.0 --cpto 2.0e4 --kpto 0.0

#### Near-resonant sea, moderate PTO (noticeable reduction in heave amplitude)

python PTOPointAbsorberSimulation.py --buoymass 500 --buoyradius 5 --waterdensity 1000 --waterdepth 100 --wavefrequency 0.20 --wavedirection 3.14159 --waveamplitude 2.0 --cpto 1.0e5 --kpto 5.0e5

#### Near-resonant sea, strong PTO (heave strongly damped / stiffened)

python PTOPointAbsorberSimulation.py --buoymass 500 --buoyradius 5 --waterdensity 1000 --waterdepth 100 --wavefrequency 0.20 --wavedirection 3.14159 --waveamplitude 2.0 --cpto 5.0e5 --kpto 1.0e6
