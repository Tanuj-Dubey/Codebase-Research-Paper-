@echo off
echo ====================================================================
echo Starting Hybrid Model Cross-Battery Ablation Study (28 Permutations)
echo ====================================================================
echo.
echo This will test all combinations of B0005, B0006, B0007, and B0018 to
echo generate the final generalization table for your IEEE paper.
echo.
echo WARNING: This process runs 28 heavy experiments back-to-back.
echo It may take 30 to 45 minutes to fully complete. 
echo As it runs, results will automatically be saved to 'cross_battery_ablation.csv'.
echo You can open that file in Excel at any time to see the live progress.
echo.
pause
python run_cross_battery_ablation.py
echo.
echo ====================================================================
echo FINISHED! The results are saved in cross_battery_ablation.csv
echo ====================================================================
pause

