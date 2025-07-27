# Structure-Aware Prototype Guided Trusted Multi-View Classification

## Motivation
Trustworthy multi-view classification (TMVC) is inherently challenging due to the heterogeneity, inconsistency, and potential conflicts that arise among different data sources. These issues not only complicate the integration of multi-view information but also compromise the reliability of classification outcomes. Existing TMVC approaches typically rely on globally dense neighbor relationships to capture intra-view dependencies, which results in high computational costs and limited scalability. More importantly, these methods often lack explicit mechanisms to enforce inter-view consistency or to ensure that the aggregated evidence reflects true class-level consensus.

To address these limitations, we propose a prototype-based TMVC framework that rethinks how multi-view neighbor structures are modeled and aligned. By representing each view’s local structure with prototypes, our approach simplifies intra-view relation modeling and reduces computation. Furthermore, the framework dynamically aligns intra- and inter-view structures within each minibatch, enabling consistent cross-view consensus building in an efficient manner. This design eliminates the need for manually assigned weights and promotes trustworthiness by ensuring that neighbor structures across views are harmonized within the class space.

Through extensive experiments on multiple public datasets, our method demonstrates improved classification performance and robustness over prevailing TMVC baselines, particularly in complex settings where view conflict and uncertainty are prominent.

## How to run?

You can run the main.py by using the following commands.
    
    python main.py

Regarding the selection of different datasets, you only need to find the corresponding data set comment lines shown below in the main function, remove the comment lines of the data sets you need to run, and comment out the unnecessary comment lines.

For example, if you want to run the PIE dataset, the following code is like this:

    #parser.add_argument('--config_file', type=str, default='configs/HandWritten.yaml') 
    parser.add_argument('--config_file', type=str, default='configs/PIE.yaml')
    #parser.add_argument('--config_file', type=str, default='configs/ALOI.yaml')

    #dataset = HandWritten()
    dataset = PIE()
    #dataset = ALOI()

If you want to run the conflict dataset, remove the comment symbol in the following code

    dataset.postprocessing(test_index, addNoise=True, sigma=0.5, ratio_noise=0.1, addConflict=True, ratio_conflict=0.4)

## Where to find the parameter configuration?

Detailed parameter configuration is in the config folder.

