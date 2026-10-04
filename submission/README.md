# Kaggle submission output

`keigo.submission.build_bundle` packages a selected candidate entrypoint and its hash
manifest as a deterministic two-file zip. It does not create agent strategy. Before
publishing a competition submission, connect an official Kaggriculture environment
adapter and validate the `agent(observation) -> action` contract.
