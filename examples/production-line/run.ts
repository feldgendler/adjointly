/* `npm run example`: the walkthrough of main.ts on the default line. */

import { initialSchedule, runExample } from './main.ts';
import { defaultLine } from './model.ts';

runExample({
  line: defaultLine,
  start: initialSchedule,
  maxShift: 5,
  maxTransfer: 3,
  seeds: [1, 2, 3, 4, 5],
  log: (line) => {
    console.log(line);
  },
});
