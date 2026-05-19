Retrieve:  
   Search for artifacts/solutions to problems/goals/tasks.  
   Multiple tools for various kinds and domains.

Compose:
   Create a new artifact/solution thru composition of sub-problems/goals/tasks.
   In addition to the control flow the identification and routing of inputs and outputs is also required.
   A functional semantics is good for keeping things simple but not the only option.

   A few common cases:
   
   TODO list:  In general form this is a DAG with each item listing the things 
   that need to complete before it is run (prerequisites).  

   Conditional: 
      IF/THEN/ELSE
      CASE SWITCH ELSE

   Loop:
      DO/WHILE
      FOR 
    
   Filter:

   Transform:

   Exception handling:

Transform:
   Transformation patterns are a little less obvious since most programming languages don't modify code.

   Generalization:  
     Change a constant in a program to a variable/input.
     For example the [When is Mardi Gras] query will generate code to compute for some year which
     might be fixed or based on the current date.  A subsequent request for a particular year 
     could take the solution for a fixed year (a literal or dynamic based on current date) and transform it to take the year as an in input.
    
