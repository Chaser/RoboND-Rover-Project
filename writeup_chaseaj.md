## Project: Search and Sample Return
## Engineer: Chase Johnson

---

**Aim:**  The aim of the `Search and Sample Return` project is to give experience of all three essential elements of robotics - `Perception`, `decision making`, and `action`. This will be achieved by receiving raw image data from a "rover" to navigate and avoid obstacticles while locating rocks.

### Notebook Analysis

#### Procedure

The `perception` element of the project consists of following three main steps:

* Color Transform
* Perspective Transform,
* Storing Map Data

**Color Transform**
Determination of naviagable terrain by isolating different color isolation and differenation of terrain objects.

**Perspective Transform**
Perspective Transform acquires the raw image data from the forward facing camera to a top map view while ensuring camera orientation is accounted for when considering co-ordinates on a static map.

**Storing Map Data**
Once the captured image is transformed its important to store this data on the world map

