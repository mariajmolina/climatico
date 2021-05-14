#!/bin/csh -fxv
#

foreach c ( file_*.nc )

  set file = $c
  set filen = iso20c_$file
  cp ~/python_scripts/climatico/isotherm/isotherm_20c_multi.jnl $c.ccsm.jnl
  sed -e "1,$ s'oldfile'$file'" $c.ccsm.jnl >! $c.ccsm1.jnl
  sed -e "1,$ s'newfile'$filen'" $c.ccsm1.jnl >! $c.ccsm.jnl
  pyferret < $c.ccsm.jnl
  echo "removing {$c.ccsm.jnl} and {$c.ccsm1.jnl}"
  rm $c.ccsm.jnl
  rm $c.ccsm1.jnl
  end
