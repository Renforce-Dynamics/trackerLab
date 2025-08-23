# Obs Construction

`dof_body_ids`: Indicates the nodes should be used in the skeleton.
`dof_offsets`: Indicates the nodes expansion to the obs subset, the final length is the obs size.
`dof_indices_sim` & `dof_indices_motion`: These indice were used for reordering the yaw, pitch, roll, generation, where the calced dof pos is `pitch, yaw, roll` which might not be same with gym's `yaw, roll, pitch`.

Generally speaking, the exp map result is `roll, pitch, yaw`, then reorder to your seq