Components
==========

TABASCAL uses a component-based model description. This means that various different models can be created where each model component can be selected freely in a configuration file. Therefore, each component should work with all others. There is a standard component class template where the base class is defined in :class:`~tabascal.components.Component`

The components pass their results to one another through a shared state dictionary, along the edges of the graph below. Each component declares the state keys it reads as ``required_inputs`` and the keys it writes as ``output_shapes``, and those declarations are checked against the configured ``model.components`` list when the model is assembled: a list that leaves out a component, or gives the right ones in the wrong order, is rejected by name before anything is computed.

.. toctree::
   :maxdepth: 1

   components/base_component
   components/trajectory
   components/rfi_signal
   components/rfi_vis
   components/ast_signal
   components/ast_vis
   components/gains

.. graphviz::

   digraph Components {
       graph [rankdir=TB, fontsize=10];
       node [shape=box, style="rounded,filled", fillcolor=white, fontcolor=black];
       edge [color=black];

       // Top-level components
       Traj [label="Trajectory\nComponent", shape=box];
       RFISig [label="RFI Signal\nComponent", shape=box];
       RFIVis [label="RFI Visibility\nCalculation\nComponent", shape=box];
       AstPos [label="Astronomical\nPosition\nComponent", shape=box];
       AstSig [label="Astronomical\nSignal\nComponent", shape=box];
       AstVis [label="Astronomical\nVisibility\nComponent", shape=box];
       Gain [label="Gain\nComponent", shape=box];

       // Data nodes
       rfi_xyz [label="• rfi_xyz\l• rfi_phase", shape=ellipse, fillcolor=lightgray, fontname="Courier"];
       rfi_A [label="rfi_A", shape=ellipse, fillcolor=lightgray, fontname="Courier"];
       vis_rfi [label="vis_rfi", shape=ellipse, fillcolor=lightgray, fontname="Courier"];
       ast_radec [label="ast_radec", shape=ellipse, fillcolor=lightgray, fontname="Courier"];
       ast_I [label="• ast_I\l• ast_shape\l", shape=ellipse, fillcolor=lightgray, fontname="Courier"];
       vis_ast [label="vis_ast", shape=ellipse, fillcolor=lightgray, fontname="Courier"];
       vis_obs [label="• gains\l• vis_obs", shape=ellipse, fillcolor=lightgray, fontname="Courier"];

       // Edges
       Traj -> rfi_xyz;
       RFISig -> rfi_A;
       rfi_xyz -> RFIVis;
       rfi_A -> RFIVis;
       RFIVis -> vis_rfi;
       AstPos -> ast_radec;
       AstSig -> ast_I;
       ast_radec -> AstVis;
       ast_I -> AstVis;
       AstVis -> vis_ast;
       vis_rfi -> Gain;
       vis_ast -> Gain;
       Gain -> vis_obs;
   }

