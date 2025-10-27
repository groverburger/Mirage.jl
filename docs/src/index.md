# Mirage.jl

Welcome to Mirage.jl! A lightweight, immediate-mode 2D & 3D rendering library for Julia.

## Overview

Mirage.jl provides a simple and productive way to create real-time graphics. It is designed for developers, researchers, and artists who want to quickly visualize algorithms, render simulations, or build custom interactive applications without the overhead of a large engine. Its API is heavily inspired by the simplicity and ease of use of the HTML5 Canvas. Mirage.jl easily composes with other libraries such as CImGui.jl to create feature-rich applications. This expressiveness, combined with Julia's REPL-driven development process, unlocks unparalleled development speed.

### Key Features

*   **Immediate-Mode API**: A simple, stateful API for 2D drawing (`fillrect`, `stroke`, `translate`, `rotate`). If you know the HTML5 Canvas, you'll feel right at home.
*   **Simple 3D Rendering**: Load `.obj` models or create custom procedural meshes like cubes and spheres. Position them in 3D space with a simple camera system.
*   **Lightweight and Cross-Platform**: Built on top of GLFW.jl and ModernGL.jl, Mirage.jl is easy to set up and runs on Windows, macOS, and Linux.
*   **State Management**: A stack-based graphics state (`save()` and `restore()`) makes it easy to manage transformations and styles in complex scenes.

### Navigation

*   **[Getting Started](getting_started.md)**: Install Mirage.jl and render your first shape in minutes.
*   **[Core Concepts](concepts.md)**: Understand the key ideas behind Mirage.jl, including the immediate-mode paradigm and the rendering pipeline.
*   **[API Reference](api_reference.md)**: A detailed breakdown of all available functions and types.
*   **[Examples](examples.md)**: Practical code for 2D and 3D scenes to get you started.
