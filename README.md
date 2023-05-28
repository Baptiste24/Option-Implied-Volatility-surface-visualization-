# Option-Implied-Volatility-surface-visualization-
Option Implied Volatility surface visualization 

![image](https://github.com/Baptiste24/Option-Implied-Volatility-surface-visualization-/assets/132922554/7719ea53-54e9-477f-8c48-c3ef89c14c25)


This tool enable anyone to visualize any option's implied volatility surface through a Dash app. 
Please save this file and go to your anaconda prompt and type "python thefilename", it should return an html link.

This highlight the vol smile and term structure. The surface is smoothed thanks to a spline interpolator.
An hyperplane is set to help visualize the level of vol.

The tool uses yahoo_fin library to get option chain data.

NOTES : the spline is rudimentary hence some surface can be sketchy, this is part of future work.
