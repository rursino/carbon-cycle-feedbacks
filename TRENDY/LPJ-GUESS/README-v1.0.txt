Trendy-v8 GCP output 2019 from LPJ-GUESS
contact: peter.anthoni@kit.edu; almut.arneth@kit.edu

S0,S1,S2,S3,S4,S5,S6,S7 (v1.0) submitted August and September, 2019

Note: LPJ-GUESS per PFT format for lai, gpppft, npppft, cVegpft, and transpft:
  The files contain the contribution of each PFT to the gridcell total values.
  For example the LPJ-GUESS_S3_lai.nc give the contribution of each PFT to the gridcell LAI value, to get the gridcell LAI all LAI_per_PFT just need to be summed. There is no need to scale by landcoverfrac per PFT and calculate the sum of the (LAI_per_PFT * landcoverfrac_per_PFT).
  The only exception are the theightpft, which provides the mean tree height for each PFT (non tree PFT are included by zeroed) and the landCoverFrac per PFT, which gives the foliage projected cover for each PFT (bare gound is not included, can be calculated as 1.0 - sum(landCoverFrac) - oceanCoverFrac)

Note: Nitrogen inputs are based on NMIP fertilization and CMIP6 nitrogen deposition (hist+ssp585) data


content:

yearly 	
cVeg 			total carbon in vegetation
cLitter		carbon in above-ground litter
cSoil			carbon in soil, incl. below-ground litter
cProduct	carbon in harvested vegetation slow pool
fFire			carbon emissions from fire
fLuc			carbon emissions from land use change, incl. clearance and wood products
fHarvest  carbon emissions from crop harvest
fGrazing	carbon emissions from pasture harvest
nbp			  carbon net biosphere production
cLeaf			carbon in leaves
cWood			carbon in wood, incl. hard-, sapwood and coarse roots
cRoot			carbon in roots, excl. coarse roots
cCwd			carbon in coarse woody debris
aburntArea	annual burnt area fraction in %
nVeg			total nitrogen in vegetation
nLitter		nitrogen in above-ground litter
nSoil			nitrogen in soil, incl. below-ground litter
nProduct	nitrogen in harvested vegetation slow pool
fNdep			nitrogen deposition
fBNF			biological nitrogen fixation
fNup      nitrogen uptake of vegetation
fNnetmin	net nitrogen mineralisation
fNloss    total ecosystem nitrogen loss
fNflux    flux part of total ecosystem nitrogen loss
fNleach   leach part of total ecosystem nitrogen loss
oceanCoverFrac time invariant fractional ocean cover (includes inland water)

yearly per pft
cVegpft			  carbon in vegetation per PFT
landCoverFrac	fractional land cover of PFT
theightpft    tree height per pft

monthly
gpp        gross primary production
npp        net primary production
ra			   autotrophic respiration
mrro			 total runoff
mrso       total soil moisture content
msl        soil moisture content upper (0-50cm) and lower (50-150cm) soil layer
tsl			   temperature of upper soil layer (0.25m below surface)
evapotrans total latent heat flux
tran			 latent heat flux from transpiration
evspsblsoi latent heat flux from bare soil evaporation
evspsblveg latent heat flux from leaf evaporation

monthly per pft
transpft	transpiration per PFT
gpppft		gross primary production per PFT, excl. establishment
npppft		net primary production per PFT, excl. establishment
lai		    leaf area index per PFT

Note that outputs were not submitted that show questionable patterns (monthly heterotrophic respiration), while others could not be obtained due to model constraints (e.g., pft-level nbp, pft-level sensible heat flux, etc.).
