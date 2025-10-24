# Core Concepts

Mirage.jl is an immediate-mode rendering library for Julia. This page explains the core concepts behind this approach.

## Immediate-Mode Rendering

Immediate-mode rendering is a paradigm where rendering commands are issued directly to the graphics API for immediate execution. The application is responsible for drawing the entire scene in every frame.

### How it Differs from Retained-Mode

In contrast, retained-mode rendering involves building a scene graph or a list of objects that the rendering engine manages and renders. The application manipulates the scene graph, and the engine handles the drawing.

### Comparison to HTML5 Canvas API

The immediate-mode approach is very similar to the HTML5 Canvas API. Both provide a low-level, procedural way to draw graphics, which can be very intuitive for developers familiar with this model.

### When to Use Immediate-Mode

*   **Prototyping and experimentation:** Quickly visualize data or algorithms.
*   **Simple 2D/3D graphics:** When you don't need the complexity of a full scene graph.
*   **Custom rendering pipelines:** When you want full control over the rendering process.
