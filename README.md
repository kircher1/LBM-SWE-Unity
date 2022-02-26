# LBM-SWE-Unity
This repository contains a Unity package with a simulator for shallow water flows, based on the D2Q9 model of the Lattice Boltzmann method.

![LBM-Loop](https://user-images.githubusercontent.com/20366429/155856962-8f38a7aa-1ea7-4dde-a7c2-93227a541edb.gif)

I wanted to try out Unity's Job System and Burst compiler, and the LBM simulation is very paralellizable i.e., it is good problem to solve with Jobs and Burst. (Compute shaders would also be a great tool.)

The algorithm and equations are explained excellently and in great detail by [Zhou, J. G. (2013). _Lattice Boltzmann Methods for Shallow Water Flows_ (2004th ed.). Springer.](https://link.springer.com/book/10.1007/978-3-662-08276-8)

# Sample Project
To see a Unity project demonstrating how to setup and visualize the LBM sim...
* Clone the repo
* Open the Unity project here: `<repo-root>/LBM-SWE-UnityProject`

# Install Package
To install the package into a Unity project...
* Follow these instructions to install via a Git URL: https://docs.unity3d.com/Manual/upm-ui-giturl.html
* Use this URL for the package: `https://github.com/kircher1/LBM-SWE-Unity.git?path=/com.kircher.lbmswe`
