# Data sources and licenses

The raw Bitbrains GWA-T-12 fastStorage and Rnd archives are **not redistributed**.
They are available from the Grid Workloads Archive and must be used under the
archive's terms. The expected SHA-256 values used in the article are:

```text
fastStorage  11313f528a0cbcbe57e63162f8ae5a41a9c7e7c1a79872e294ff3c5bbaa2e671
Rnd          d3d9ddebb689c0b5463f2e4cfd8956e84bdcdf138b4476320855393e2b229a06
```

The provider requires acknowledgement of Bitbrains in published material,
citation of the associated CCGrid 2015 paper, and asks users to consider
acknowledging the Grid Workloads Archive.

The two small files in `data_external/` are 300-second whole-datacenter
aggregate derivatives from the Google 2019 and Alibaba 2018 public traces.
They were obtained from *DataCenter-Traces-Datasets, version 2* (Zenodo DOI
`10.5281/zenodo.14564935`), distributed under CC BY 4.0. Attribution remains
due to the original providers and dataset curators. They are supplied only to
make the reported cross-trace calculations directly reproducible.

Provider sources:
- Google ClusterData2019: https://github.com/google/cluster-data
- Alibaba cluster-trace-v2018: https://github.com/alibaba/clusterdata
- Derived aggregate archive: https://doi.org/10.5281/zenodo.14564935
