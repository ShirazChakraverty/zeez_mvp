
# Zeez 😴

_sleep improvement using wearables and n-of-1 experiments_

## “Merging Biometric Telemetry with N-of-1 Randomized Experiments to Personalize Recommendations across Individuals, Time, and Actionable Interventions - an Application to Sleep Quality Improvement”

### By:
- [Shiraz Chakraverty](https://www.linkedin.com/in/shirazchakraverty/, "LinkedIn Profile")
- [Ed Gelman](https://www.linkedin.com/in/edgelman/, "LinkedIn Profile")
- [Marcelo Queiroz](https://www.linkedin.com/in/mscatolinqueiroz/, "LinkedIn Profile")
- [Peter Trenkwalder](https://www.linkedin.com/in/peter-trenkwalder-b2959a9/, "LinkedIn Profile")

### Project Links

- [Repository](https://github.com/ShirazChakraverty/zeez_mvp)
- [Demo](https://bioloopsleep.com/demo)
- [Website](http://people.ischool.berkeley.edu/~marcelo.queiroz/Zeez/)

### Overview

As wearable biometric sensors become ubiquitous, and self-experimentation becomes more popular as a means of improving quality of life, the ability to generate evidence-based lifestyle recommendations using these data streams becomes a technological reality. Meanwhile, [new causal inference techniques](https://www.pnas.org/content/113/27/7353) are being developed for within-individual datasets and estimating heterogeneous treatment effects in traditional randomized controlled trial (RCT) settings. Currently missing from this landscape is the ability to generate insights from unbiased sensor-telemetry, in diverse populations, with many simultaneously experimentally manipulated treatments.

Such efforts were not possible in the past due to several limitations: the observational nature of most existing telemetry data sets do not allow for strong causal claims, and meanwhile, randomized controlled trials focus on estimating an “average treatment effect” for a “representative” population, leaving actionable insights “for an individual, today” far from the focus. Personalization to the individual and their current state is, therefore, a new frontier for driving value for the individual.  

Our solution builds on the recommendations made in the journal of [Personalized Medicine](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3118090/), which emphasizes the opportunities that become available once unbiased outcomes (not self-reported, which are subject to many more biases) can be measured within individuals with randomized intervention schedules. Our method extends this concept to be flexible across _many_ types of interventions randomized across a large population.

This blog describes a technical implementation of a recommendation system that attempts to address these problems. By merging many tens of thousands of days of detailed telemetry from a diverse user population and within-individual randomized controlled experiments conducted within the same user base, we are able to provide evidence-based lifestyle recommendations that account for both within- and across-individual variability. We provide an “Perturbation-Estimated Treatment Effect” (PETE) for multiple self-guided lifestyle changes - such as exercise timing and melatonin supplementation - culminating in an informed suggestion to the user meant to help them improve their sleep quality.

### How it Works

This recommender system is built by composing 4 components:
1. Many users wear Oura rings, which record sleep quality in an unbiased fashion. Some of these users run randomized experiments with lifestyle changes.
2. Create a historical data set from Oura telemetry and Bioloop experiments
3. Train a machine learning model to predict average sleep score for the next 7 days
4. Compare baseline predictions to "counterfactual" predictions under simulated "versions" of the user's recent history, as described in [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/counterfactual.html)

Steps 1 through 3 are fairly routine; therefore, detailed review of the methodology is left as a [code-review exercise](https://github.com/ShirazChakraverty/zeez_mvp) to the reader. Key facts about the dataset are summarized here:

![Key Training Facts](https://github.com/ShirazChakraverty/zeez_mvp/blob/master/deliverables/figures/zeez_key_facts_training.png)

The Counterfactual prediction described in Step 4, however, requires more description. The mechanics are detailed in the following figure:

![Zeez Counterfactual Recommendation Mechanics](https://github.com/ShirazChakraverty/zeez_mvp/blob/master/deliverables/figures/zeez_counterfactual_mechanics.png)

  Where the set of recommendations is taken as those counterfactual perturbations that produced the largest increase in the predicted sleep score.

### Tradeoffs and Limitations

In traditional machine learning projects, it's fairly common to spend a large amount of time and effort optimizing for accuracy. However, in the context of creating counterfactual recommendations, it was clear that as the model was trained on more covariates and became more structurally complex - a random forest constructed with >500 trees, for example - the difference between the baseline prediction and the perturbed prediction became increasingly diluted, often to nearly zero. While it's theoretically possible that the interventions were largely ineffective and the model was correctly sending the effect sizes towards zero as it become more accurate, we believe this effect is caused by collinear features tending to "soak up" variation rightly attributable to simulated perturbations. In light of this effect, we preferred a simpler model with fewer included covariates that still outperformed a naive baseline. Precise figures can be located in the [modeling notebooks and data dictionary](https://github.com/ShirazChakraverty/zeez_mvp).

Accuracy and inference would likely be improved in the presence of additional data. Luckily, most Oura ring users sync every day as a side effect of interacting with the Oura app, and a subset of users actively runs self-experiments. In either case, the Bioloop platform is able to deliver up-to-date data for model retraining, updating, and improving the capabilities. As more experimental data becomes available,  detailed estimates can be made across more experiment types, such as intermittent fasting, wearing blue-light-blocking lenses, and others available on the Bioloop platform.

[Visit us to participate!](https://bioloopsleep.com/)

### Why it Rocks

The methodology proposed here has several contributions which we believe to be novel:
- it automatically combines experimental treatment schedules with observational telemetry on a moderately sized population (~100, and growing!)
- it simultaneously estimates the experimental effect of several concurrent interventions, aggregating across many types and particular instances of n-of-1 trials
- it is productized, that is: built with real data given with informed consent from active platform users, and is deployable in a "real" environment

Overall, we believe that the Zeez model is a promising first step towards being able to merge observational telemetry and randomized n-of-1 experiments in order to be able to give sound recommendations for improving health and wellness.


<center>

_From all of us at Zeez and Bioloop,_

_sleep well! 😴_  

</center>
