# MasterThesis-Shape-Optimisation-Using-DRL
This thesis focuses on optimising the shape of a potential curve within a microstrip arrangement environment, following Thomson’s theorem of minimum energy, where the optimal potential curve will have the least energy within the environment. A near-global solution was achieved by navigating through the microstrip arrangement environment using reinforcement learning (RL). Through various experiments, RL has proved to be a transformative method that enabled exploration within an unknown environment. In addition to RL, a genetic algorithm (GA) was employed to optimise the RL-predicted solution further and achieve the global solution.

## App Demo
https://github.com/pvmodayil/MasterThesis-Shape-Optimisation-Using-DRL/assets/66408212/05c16f81-5aee-4642-80c8-4a7e53b29277

## Contents
```
.
├── GA_optimisation         # Contains code for Genetic Algorithms optimisation of RL-predicted curves
├── Method1                 # Contains Scatter Point G point generation trial
├── Method2                 # Contains Decaying Exponential function G point generation trial
├── Method3                 # Contains Bezier Curves G point generation with threshold reward trial
├── Method4                 # Contains Bezier Curves G point generation with scaled reward trial
├── Report                  # Contains Latex code for the final report
├── streamlit-app           # Contains code for streamlit app that generates the potential curve for given board parameters
├── MasterThesis_Presentation.pptx
└── README.md
```
