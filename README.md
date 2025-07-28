# Structure-Aware Prototype Guided Trusted Multi-View Classification

## Motivation
Trustworthy multi-view classification (TMVC) is inherently challenging due to the heterogeneity, inconsistency, and potential conflicts that arise among different data sources. These issues not only complicate the integration of multi-view information but also compromise the reliability of classification outcomes. Existing TMVC approaches typically rely on globally dense neighbor relationships to capture intra-view dependencies, which results in high computational costs and limited scalability. More importantly, these methods often lack explicit mechanisms to enforce inter-view consistency or to ensure that the aggregated evidence reflects true class-level consensus.

To address these limitations, we propose a prototype-based TMVC framework that rethinks how multi-view neighbor structures are modeled and aligned. By representing each view’s local structure with prototypes, our approach simplifies intra-view relation modeling and reduces computation. Furthermore, the framework dynamically aligns intra- and inter-view structures within each minibatch, enabling consistent cross-view consensus building in an efficient manner. This design eliminates the need for manually assigned weights and promotes trustworthiness by ensuring that neighbor structures across views are harmonized within the class space.

Through extensive experiments on multiple public datasets, our method demonstrates improved classification performance and robustness over prevailing TMVC baselines, particularly in complex settings where view conflict and uncertainty are prominent.

## How to run?

You can run training on each dataset with or without postprocessing (noise and conflict) by specifying the config file. 

1. PIE Dataset (Normal) Train on the PIE dataset without adding conflicts or noise: 

    ```
   python main.py --config_file configs/PIE.yaml
    ```
   
2. PIE Dataset (Conflict) Train on the PIE dataset with conflict and noise postprocessing enabled:
  
   ```
   python main.py --config_file configs/PIE_conflict.yaml
   ```

3. HandWritten Dataset (Normal) Train on the HandWritten dataset without conflict/noise: 
    ```
    python main.py --config_file configs/HandWritten.yaml
    ```
4. HandWritten Dataset (Conflict) Train on the HandWritten dataset with conflict and noise postprocessing: 
    ``` 
    python main.py --config_file configs/HandWritten_conflict.yaml`
    ```
5. ALOI Dataset (Normal) Train on the ALOI dataset without conflict/noise: 
    ```
    python main.py --config_file configs/ALOI.yaml
    ```
6. ALOI Dataset (Conflict) Train on the ALOI dataset with conflict and noise postprocessing: 
    ```
    python main.py --config_file configs/ALOI_conflict.yaml
    ```

