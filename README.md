
# fitting a mesh pytorch3d

Takes several images of an object using renderers built into pytorch3d: a renderer with phong shading, and a silhouette renderer

In the deformation loop, we reconstruct the object by comparing against the target images. We render our intermediate mesh & texture with
the usual rgb renderer with phong shading.

<img src="output.gif" alt="My GIF" width="700"/>
