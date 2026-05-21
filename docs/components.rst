Components
==========

TABASCAL uses a component-based model description. This means that various different models can be created where each model component can be selected freely in a configuration file. Therefore, each component should work with all others. There is a standard component class template where the base class is defined in :class:`~tabascal.components.Component` 

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
       AstSig [label="Astronomical\nSignal\nComponent", shape=box];
       AstVis [label="Astronomical\nVisibility\nComponent", shape=box];
       Gain [label="Gain\nComponent", shape=box];

       // Data nodes
       rfi_xyz [label="• rfi_xyz\l• rfi_phase", shape=ellipse, fillcolor=lightgray, fontname="Courier"];
       rfi_A [label="rfi_A", shape=ellipse, fillcolor=lightgray, fontname="Courier"];
       vis_rfi [label="vis_rfi", shape=ellipse, fillcolor=lightgray, fontname="Courier"];
       ast_sky [label="• ast_radec\l• ast_I\l• ast_image", shape=ellipse, fillcolor=lightgray, fontname="Courier"];
       vis_ast [label="vis_ast", shape=ellipse, fillcolor=lightgray, fontname="Courier"];
       vis_obs [label="• gains\l• vis_obs", shape=ellipse, fillcolor=lightgray, fontname="Courier"];

       // Edges
       Traj -> rfi_xyz;
       RFISig -> rfi_A;
       rfi_xyz -> RFIVis;
       rfi_A -> RFIVis;
       RFIVis -> vis_rfi;
       AstSig -> ast_sky;
       ast_sky -> AstVis;
       AstVis -> vis_ast;
       vis_rfi -> Gain;
       vis_ast -> Gain;
       Gain -> vis_obs;
   }

