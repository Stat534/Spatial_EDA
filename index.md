EDA demo
================

# Scallops Data Example

1.  Create a test and training set from the scallop dataset. Then create
    a visual to show the test / training datasets.

<!-- end list -->

``` r
data(scallop)
scallop <- scallop %>% mutate(log.catch = log(tot.catch + 1), 
                              id = 1:n())
```

2.  Write code to estimate a non-spatial model using just the mean
    structure. Then construct a figure that includes mean predictions
    for each site in the test dataset.

3.  Now fit a model with spatial structure and construct a figure that
    includes mean predictions for each site in the test dataset.

4.  Compare the predictive ability of the spatial and non-spatial model

### Other model fitting options

  - `krige()` in the `gstat` package. However, the spatial structure
    needs to be extracted from fitting a variogram to the empirical
    variogram.

  - `geoR` has `krige.conv()` and `krige.bayes()` options see textbook
    for more.

  - `spBayes` is another package with quite a few options
