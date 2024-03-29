<html><head>
<meta name="Author" content="Anthony R. Cassandra"><title>The POMDP Page</title>



<link title="POMDP Style Sheet" rel="stylesheet" href="pomdpFileFormat_files/pomdp-style.css" type="text/css" media="all">

<header></header></head>
<body>

<p>
<table class="header">
  <tbody><tr>
    <td> <img src="pomdpFileFormat_files/pomdp.gif" height="50"></td>
    <td class="header">Partially Observable Markov Decision Processes</td>
  </tr>
</tbody></table>

</p><center><font size="-1"><b>
<a href="http://cassandra.org/pomdp/tutorial/index.shtml">Tutorial</a>
| <a href="http://cassandra.org/pomdp/papers/index.shtml">Papers</a>
| <a href="http://cassandra.org/pomdp/talks/index.shtml">Talks</a>
| <a href="http://cassandra.org/pomdp/code/index.shtml">Code</a>
| <a href="http://cassandra.org/pomdp/examples/index.shtml">Repository</a>
</b></font></center>


<p>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<a href="http://cassandra.org/pomdp/code/index.shtml">
<img src="pomdpFileFormat_files/pomdp-solve.gif">
</a>

</p><center><h1>Input POMDP File Format</h1>
</center>

<p>
Description of the file format for example files and which is used by
the 'pomdp-solve' program.

</p><p><i>For a more detailed formal specification of the syntax of this
file format see the '<a href="http://cassandra.org/pomdp/code/pomdp-file-grammar.shtml">POMDP grammar
file</a>'. There are some semantics to the format and these are discussed
in this file.</i>

</p><p>All floating point number must be specified with at least one digit
before and one digit after the decimal point.

</p><p>Comments: Everything from a '#' symbol to the end-of-line is treated
as a comment. They can appear anywhere in the file.

</p><p>The following 5 lines must appear at the beginning of the file.
They may appear in any order as long as they preceed all specifications
of transition probabilities, observation probabilities and rewards.

</p><p><b><font face="Courier New,Courier"><font size="+3">discount: %f</font></font></b>
<br><b><font face="Courier New,Courier"><font size="+3">values: [ reward,
cost ]</font></font></b>
<br><b><font face="Courier New,Courier"><font size="+3">states: [ %d, &lt;list
of states&gt; ]</font></font></b>
<br><b><font face="Courier New,Courier"><font size="+3">actions: [ %d, &lt;list
of actions&gt; ]</font></font></b>
<br><b><font face="Courier New,Courier"><font size="+3">observations:
[ %d, &lt;list of observations&gt; ]</font></font></b>

</p><p>The definition of states, actions and/or observations can be either
a number indicating how many there are or it can be a list of strings,
one for each entry. These mnemonics cannot begin with a digit.
For instance, both:

</p><p><b><font face="Courier New,Courier"><font size="+3">actions:4</font></font></b>
<br><b><font face="Courier New,Courier"><font size="+3">actions: north south
east west</font></font></b>

</p><p>will result in 4 actions being defined. The only difference is
that, in the latter, the actions can then be referenced in this file
by the mnemonic name. Even when mnemonic names are used, later
references can use a number as well, though it must correspond to the
positional numbering starting with 0 in the list of strings. The
numbers are assigned consecutively from left to right in the listing
starting with zero. 

</p><p>When listing states, actions or observations one or more whitespace
characters are the delimiters (space, tab or newline). When a number
is given instead of an enumeration, the individual elements will be referred
to by consecutive integers starting at 0.

</p><p>After the preamble, there is the optional specification of the
starting state. (Note that this is ignored for some exact solution
algorithms.) There are a number of different formats for the starting
state. You can either:
</p><ul>
  <li> enumerate the probabilities for each state,</li>
  <li> specify a single starting state, </li>
  <li> give a uniform distribution over states, or</li>
  <li> give a uniform distribution over a subset of states.</li>
</ul>
For the last one, you can either specific a list of states too be
included, or s list of states to be excluded.  Examples of this are:
<pre>   start: 0.3 0.1 0.0 0.2 0.5

   start: uniform

   start: first-state

   start: 5

   start include: first-state third state

   start include: 1 3

   start exclude: fifth-state seventh-state
</pre>

<p>After the initial five lines and optional starting state, the
speciifications of transition probabilities, observation probabilities
and rewards appear.  These specifications may appear in any order and
can be intermixed.  Any probabilities or rewards not specified in the
file are assumed to be zero.

</p><p>You may also specify a particular probability or reward more than once.
The definition that appears last in the file is the one that will take
affect. This is convenient for specifying exceptions to a more general
specification.

</p><p>To specify an individual transition probability:

</p><p><b><font face="Courier New,Courier"><font size="+3">T: &lt;action&gt; : &lt;start-state&gt;
: &lt;end-state&gt; %f</font></font></b>

</p><p>Anywhere an action, state or observation can appear, you can also put
the wildcard character '*' which means that this is true for all possible
entries that could appear here. For example:

</p><p><b><font face="Courier New,Courier"><font size="+2">T: 5 : * : 0 1.0</font></font></b>

</p><p>is interpreted as action 5 always moving the system state to state 0,
no matter what the starting state was (i.e., for all possible starting
states.)

</p><p>To specify a single row of a particular action's transition matrix:

</p><p><b><font face="Courier New,Courier"><font size="+3">T: &lt;action&gt; : &lt;start-state&gt;</font></font></b>
<br><b><font face="Courier New,Courier"><font size="+3">%f %f ... %f</font></font></b>

</p><p>Where there is an enter for each possible next state. This allows
defining the specific transition probabilities for a particular
starting state only.  Instead of a list of probabilities the mnemonic
word 'uniform' may appear. In this case, each transition for each next
state will be assigned the probability 1/#states. Again, an asterick
in either the action or start-state position will indicate all
possible entries that could appear in that position.

</p><p>To specify an entire transition matrix for a particular action:

</p><p><b><font face="Courier New,Courier"><font size="+3">T: &lt;action&gt;</font></font></b>
<br><b><font face="Courier New,Courier"><font size="+3">%f %f ... %f</font></font></b>
<br><b><font face="Courier New,Courier"><font size="+3">%f $f ... %f</font></font></b>
<br><b><font face="Courier New,Courier"><font size="+3">...</font></font></b>

</p><p><b><font face="Courier New,Courier"><font size="+3">%f $f ... %f</font></font></b>

</p><p>Where each row corresponds to one of the start states and each
column specifies one of the ending states. Each entry must be
separated from the next with one or more white-space characters. The
state numbers go from left to right for the ending states and top to
bottom for the starting states. The new-lines are just for formatting
convenience and do not affect final matrix results. The only
restriction is there must be NxN values specified where 'N' is the
number of states.

</p><p>In addition, there are a few mnemonic conventions that can be used in
place of the explicit matrix:

</p><p><b><font face="Courier New,Courier"><font size="+3">identity</font></font></b>
<br><b><font face="Courier New,Courier"><font size="+3">uniform</font></font></b>

</p><p>Note that uniform means that each row of the transition matrix will
be set to a uniform distribution. The identity mnemonic will result
in a transition matrix that leaves the underlying state unchanged for all
possible starting states (i.e., the identity matrix).

</p><p>The observational probabilities are specified in a maner similiar to
the transition probabilities. To specify individual observation probabilities:

</p><p><b><font face="Courier New,Courier"><font size="+3">O : &lt;action&gt; :
&lt;end-state&gt; : &lt;observation&gt; %f</font></font></b>

</p><p>The asterick wildcard is allowed in any of the positions.

</p><p>To specify a row of a particular actions observation probability
<br>matrix:

</p><p><b><font face="Courier New,Courier"><font size="+3">O : &lt;action&gt; :
&lt;end-state&gt;</font></font></b>
<br><b><font face="Courier New,Courier"><font size="+3">%f %f ... %f</font></font></b>

</p><p>This specifies a probability of observing each possible observation
for a particular action and ending state. The mnemonic short-cut
'uniform' may also appear in this place.

</p><p>To specify an entire observation probability matrix for an action:

</p><p><b><font face="Courier New,Courier"><font size="+3">O: &lt;action&gt;</font></font></b>
<br><b><font face="Courier New,Courier"><font size="+3">%f %f ... %f</font></font></b>
<br><b><font face="Courier New,Courier"><font size="+3">%f $f ... %f</font></font></b>
<br><b><font face="Courier New,Courier"><font size="+3">...</font></font></b>

</p><p><b><font face="Courier New,Courier"><font size="+3">%f $f ... %f</font></font></b>

</p><p>The format is similiar to the transition matrices except the number
of entries must be NxO where 'N' is the number of states and 'O' is the
number of observations. Here too the 'uniform' mnemonic can be substituted
for an enire matrix. In this case it will assign each entry of each row
the probability 1/#observations.

</p><p>To specify individual rewards:

</p><p><b><font face="Courier New,Courier"><font size="+3">R: &lt;action&gt; : &lt;start-state&gt;
: &lt;end-state&gt; : &lt;observation&gt; %f</font></font></b>

</p><p>For any of the entries, an asterick for either <b><font face="Courier New,Courier"><font size="+3">&lt;state&gt;</font></font></b>,
<b><font face="Courier New,Courier"><font size="+3">&lt;action&gt;</font></font></b>,
<b><font face="Courier New,Courier"><font size="+3">&lt;observation&gt;</font></font></b>
indicates a wildcard that will be expanded to all existing entities.

</p><p>There are two other forms to specify rewards:

</p><p><b><font face="Courier New,Courier"><font size="+3">R: &lt;action&gt; : &lt;start-state&gt;
: &lt;end-state&gt;</font></font></b>
<br><b><font face="Courier New,Courier"><font size="+3">%f %f ... %f</font></font></b>

</p><p>This specifies a particular row of a reward matrix for a particular
action and start state. The last reward specification form is

</p><p><b><font face="Courier New,Courier"><font size="+3">R: &lt;action&gt; : &lt;start-state&gt;</font></font></b>
<br><b><font face="Courier New,Courier"><font size="+3">%f %f ... %f</font></font></b>
<br><b><font face="Courier New,Courier"><font size="+3">%f %f ... %f</font></font></b>
<br><b><font face="Courier New,Courier"><font size="+3">...</font></font></b>
<br><b><font face="Courier New,Courier"><font size="+3">%f %f ... %f</font></font></b>

</p><p>which lets you specify an entire reward matrix for a particular action
and start-state combination.

</p><p>


</p><p>
</p><p>
</p><hr>
<table class="footer">
  <tbody><tr>
    <td class="footer"> � 2003-2005, Anthony R. Cassandra</td>
    <td align="right"><img src="pomdpFileFormat_files/pomdp.gif" height="25"></td>
  </tr>
</tbody></table>


<!-- hhmts start -->
Last modified: Tue Mar  1 11:06:37 CST 2005
<!-- hhmts end --><br>
</body></html>
