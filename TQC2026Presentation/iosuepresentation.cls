\ProvidesClass{iosuepresentation}[2021/04/05 Simple class for presentations]
\usepackage{amsmath,amssymb,amsthm,amsfonts,graphicx,mathtools}
\usepackage{cmap} % Makes ligatured fonts searchable and copyable in pdf readers

% \graphicspath{ {./graphics/} }

% to allow for defining commands with *
% e.g. see \note and \note* below
\usepackage{suffix}

% any options that are provided to iosuepresentation
% go directly to beamer
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{beamer}}
\ProcessOptions\relax
% possible options are handout; handout,notes; handout,notes=only.
% use note outside of frame.  \note{Stuff} or \note[itemize]{\item Stuff.}

\LoadClass[aspectratio=169]{beamer}

\usepackage{bm}

\let\oldappendix\appendix
\renewcommand{\appendix}{
  \oldappendix
  \section{Additional slides}
  \begin{frame}[noframenumbering,plain,c]
    \begin{center}
      \Huge Additional slides
    \end{center}
  \end{frame}
}

% get rid of hyperref warning
\renewcommand{\appendixname}{\texorpdfstring{\translate{Appendix}}{Appendix}}

% meta
\newcommand{\iosuename}{Joseph~T.~Iosue}
\newcommand{\iosueemail}{\href{mailto:jtiosue@umd.edu}{jtiosue@umd.edu}}
\newcommand{\UMD}{University~of~Maryland,~College~Park,~Maryland~20742,~USA}
\newcommand{\JQI}{Joint~Quantum~Institute, NIST/\UMD}
\newcommand{\QUICS}{Joint~Center~for~Quantum~Information~and~Computer~Science, NIST/\UMD}

% bb
\newcommand{\bbA}{\mathbb{A}}
\newcommand{\bbB}{\mathbb{B}}
\newcommand{\bbC}{\mathbb{C}}
\newcommand{\bbD}{\mathbb{D}}
\newcommand{\bbE}{\mathbb{E}}
\newcommand{\bbF}{\mathbb{F}}
\newcommand{\bbG}{\mathbb{G}}
\newcommand{\bbH}{\mathbb{H}}
\newcommand{\bbI}{\mathbb{I}}
\newcommand{\bbJ}{\mathbb{J}}
\newcommand{\bbK}{\mathbb{K}}
\newcommand{\bbL}{\mathbb{L}}
\newcommand{\bbM}{\mathbb{M}}
\newcommand{\bbN}{\mathbb{N}}
\newcommand{\bbO}{\mathbb{O}}
\newcommand{\bbP}{\mathbb{P}}
\newcommand{\bbQ}{\mathbb{Q}}
\newcommand{\bbR}{\mathbb{R}}
\newcommand{\bbS}{\mathbb{S}}
\newcommand{\bbT}{\mathbb{T}}
\newcommand{\bbU}{\mathbb{U}}
\newcommand{\bbV}{\mathbb{V}}
\newcommand{\bbW}{\mathbb{W}}
\newcommand{\bbX}{\mathbb{X}}
\newcommand{\bbY}{\mathbb{Y}}
\newcommand{\bbZ}{\mathbb{Z}}

% cal
\newcommand{\calA}{\mathcal{A}}
\newcommand{\calB}{\mathcal{B}}
\newcommand{\calC}{\mathcal{C}}
\newcommand{\calD}{\mathcal{D}}
\newcommand{\calE}{\mathcal{E}}
\newcommand{\calF}{\mathcal{F}}
\newcommand{\calG}{\mathcal{G}}
\newcommand{\calH}{\mathcal{H}}
\newcommand{\calI}{\mathcal{I}}
\newcommand{\calJ}{\mathcal{J}}
\newcommand{\calK}{\mathcal{K}}
\newcommand{\calL}{\mathcal{L}}
\newcommand{\calM}{\mathcal{M}}
\newcommand{\calN}{\mathcal{N}}
\newcommand{\calO}{\mathcal{O}}
\newcommand{\calP}{\mathcal{P}}
\newcommand{\calQ}{\mathcal{Q}}
\newcommand{\calR}{\mathcal{R}}
\newcommand{\calS}{\mathcal{S}}
\newcommand{\calT}{\mathcal{T}}
\newcommand{\calU}{\mathcal{U}}
\newcommand{\calV}{\mathcal{V}}
\newcommand{\calW}{\mathcal{W}}
\newcommand{\calX}{\mathcal{X}}
\newcommand{\calY}{\mathcal{Y}}
\newcommand{\calZ}{\mathcal{Z}}

% brackets
\newcommand{\parentheses}[1]{\left(#1\right)}
\newcommand{\brackets}[1]{\left[#1\right]}
\newcommand{\curlybrackets}[1]{\left\{#1\right\}}
\newcommand{\set}{\curlybrackets}
\newcommand{\mset}[1]{\left\{\!\!\left\{ #1 \right\} \!\! \right\}}
\newcommand{\pargs}[1]{\!\parentheses{#1}}
\newcommand{\bargs}[1]{\!\brackets{#1}}
\newcommand{\cbargs}[1]{\!\curlybrackets{#1}}
\newcommand{\angles}[1]{\left\langle #1\right\rangle}
\newcommand{\ceil}[1]{\left\lceil #1\right\rceil}
\newcommand{\floor}[1]{\left\lfloor #1\right\rfloor}
\newcommand{\abs}[1]{\left\lvert #1 \right\rvert}
\newcommand{\norm}[1]{\left\lVert #1 \right\rVert}
% with *, don't automatically size
\WithSuffix\newcommand\parentheses*[1]{(#1)}
\WithSuffix\newcommand\brackets*[1]{[#1]}
\WithSuffix\newcommand\curlybrackets*[1]{\{#1\}}
\WithSuffix\newcommand\mset*[1]{\{\!\!\{ #1 \} \!\!\}}
\WithSuffix\newcommand\pargs*[1]{(#1)}
\WithSuffix\newcommand\bargs*[1]{[#1]}
\WithSuffix\newcommand\cbargs*[1]{\{#1\}}
\WithSuffix\newcommand\angles*[1]{\langle #1\rangle}
\WithSuffix\newcommand\ceil*[1]{\lceil #1\rceil}
\WithSuffix\newcommand\floor*[1]{\lfloor #1\rfloor}
\WithSuffix\newcommand\abs*[1]{\lvert #1 \rvert}
\WithSuffix\newcommand\norm*[1]{\lVert #1 \rVert}

% brakets
\newcommand{\ket}[1]{\@ifnextchar\bra{\mathinner{\lvert#1\rangle}\!}{\mathinner{\lvert#1\rangle}}}
\newcommand{\bra}[1]{\@ifnextchar\ket{\mathinner{\langle#1}\!}{\mathinner{\langle#1\rvert}}}
\newcommand{\Ket}[1]{\@ifnextchar\Bra{\mathinner{\big\lvert#1\big\rangle}\!}{\mathinner{\big\lvert#1\big\rangle}}}
\newcommand{\Bra}[1]{\@ifnextchar\Ket{\mathinner{\big\langle#1}\!}{\mathinner{\big\langle#1\big\rvert}}}

% other
\DeclareMathOperator*{\argmax}{arg\!\max}
\DeclareMathOperator*{\argmin}{arg\!\min}
\newcommand{\Dom}[1]{\operatorname{Dom}\pargs{#1}}
\newcommand{\sgn}[1]{\operatorname{sgn}\pargs{#1}}
\newcommand{\Tr}{\operatorname{Tr}}
\newcommand{\dd}{\mathop{}\!\mathrm{d}}
\newcommand{\Dd}[1]{\mathop{}\!\mathrm{d^#1}}

% probability
\DeclareMathOperator*{\Expval}{\mathbb{E}}
\newcommand{\EX}[2]{\Expval_{#1}\bargs{#2}}
\DeclareMathOperator*{\Prob}{\operatorname{Pr}}
\renewcommand{\Pr}[2]{\Prob_{#1}\bargs{#2}}

% morphisms
\newcommand{\Hom}[1]{\operatorname{Hom}\pargs{#1}}
\newcommand{\End}[1]{\operatorname{End}\pargs{#1}}
\newcommand{\Aut}[1]{\operatorname{Aut}\pargs{#1}}

% big O notation
% big O notation
\newcommand{\bigO}[1]{\calO\pargs{#1}}
\newcommand{\littleo}[1]{o\pargs{#1}}
\newcommand{\bigOmega}[1]{\Omega\pargs{#1}}
\newcommand{\littleomega}[1]{\omega\pargs{#1}}
\newcommand{\bigTheta}[1]{\Theta\pargs{#1}}
\newcommand{\poly}[1]{\operatorname{poly}\pargs{#1}}
\newcommand{\polylog}[1]{\operatorname{polylog}\pargs{#1}}
% with *, don't automatically size
\WithSuffix\newcommand\bigO*[1]{\calO\pargs*{#1}}
\WithSuffix\newcommand\littleo*[1]{o\pargs*{#1}}
\WithSuffix\newcommand\bigOmega*[1]{\Omega\pargs*{#1}}
\WithSuffix\newcommand\littleomega*[1]{\omega\pargs*{#1}}
\WithSuffix\newcommand\bigTheta*[1]{\Theta\pargs*{#1}}
\WithSuffix\newcommand\poly*[1]{\operatorname{poly}\pargs*{#1}}
\WithSuffix\newcommand\polylog*[1]{\operatorname{polylog}\pargs*{#1}}

% more math
\newcommand{\e}{\mathrm{e}}
\renewcommand{\i}{\mathrm{i}}
\newcommand{\U}{\mathrm{U}}
\renewcommand{\O}{\mathrm{O}}
\newcommand{\Sp}{\mathrm{Sp}}
\newcommand{\GL}{\mathrm{GL}}
\renewcommand{\Re}{\mathrm{Re}}
\renewcommand{\Im}{\mathrm{Im}}


\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows}
\tikzstyle{arrow} = [thick,->,>=stealth]
\newcommand{\absolutePos}[3]{
  \begin{tikzpicture}[remember picture,overlay]
    \node[xshift=#1,yshift=-#2] at (current page.north west){#3};
  \end{tikzpicture}
}
\usetikzlibrary{decorations.pathreplacing,patterns,math,external,decorations.pathmorphing,arrows.meta}



%%% t-Gaussian states
\usepackage{xspace}

% \newcommand{\G}[1]{\( (#1) \)-Gaussian\xspace}
% \WithSuffix\newcommand\G*[1]{\( #1 \)-Gaussian\xspace}

\newcommand{\G}[1]{\( G_{#1} \)\xspace}
\WithSuffix\newcommand\G*[1]{\G{#1}}


\newcommand{\Gt}{\texorpdfstring{\G{t}}{Gt}}
\newcommand{\Gtt}{\texorpdfstring{\G{2t}}{G2t}}
% \newcommand{\Gt}{\G{t}}
% \newcommand{\Gtt}{\G{2t}}



% The Beamer class comes with a number of default slide themes
% which change the colors and layouts of slides. Below this is a list
% of all the themes, uncomment each in turn to see what they look like.

\usetheme{Boadilla}
% \usetheme{CambridgeUS}
% \usetheme{Madrid}
% \usetheme{Warsaw} 
% \usetheme{metropolis}  

% As well as themes, the Beamer class has a number of color themes
% for any slide theme. Uncomment each of these in turn to see how it
% changes the colors of your current slide theme.

% https://hartwork.org/beamer-theme-matrix/
\usecolortheme{beaver}

% Change title background
\setbeamercolor{title}{bg=white}

% to set the headline
\setbeamertemplate{headline}{}

% count the main frames and appendix frames differently
\setbeamertemplate{page number in head/foot}[appendixframenumber]

% To remove the footer line in all slides uncomment this line
%\setbeamertemplate{footline}

% To replace the footer line in all slides with a simple slide count uncomment this line
%\setbeamertemplate{footline}[page number]

% To replace the footer line in all slides with a simple frame count uncomment this line
% \setbeamertemplate{footline}[frame number]

% otherwise, this changes the background color of the footline
% https://github.com/josephwright/beamer/blob/main/base/themes/color/beamercolorthemebeaver.sty
\setbeamercolor{palette tertiary}{fg=darkred!60!black,bg=gray!30!white}



\setbeamertemplate{navigation symbols}{} % To remove the navigation symbols from the bottom of all slides uncomment this line

% space for bulleted and numbered lists.
\newenvironment{spaceditemize}[1][1em]{\begin{itemize}\setlength{\itemsep}{#1}}{\end{itemize}}
\newenvironment{spacedenumerate}[1][1em]{\begin{enumerate}\setlength{\itemsep}{#1}}{\end{enumerate}}

\newenvironment{spacedpauseditemize}[1][1em]{\begin{itemize}[<+->]\setlength{\itemsep}{#1}}{\end{itemize}}
\newenvironment{spacedpausedenumerate}[1][1em]{\begin{enumerate}[<+->]\setlength{\itemsep}{#1}}{\end{enumerate}}

% footnote with no footnote mark
\makeatletter
\def\blfootnote{\gdef\@thefnmark{}\@footnotetext}
\makeatother

% show frame with table of contents
% optional argument can be empty, currentsection
% currentsubsection, etc.
\newcommand{\overviewslide}[1][]{
  \begin{frame}{Overview}
    \tableofcontents[#1]
  \end{frame}
}

% \AtEndDocument{
%   \begin{frame}[noframenumbering,plain,c]
%     % \begin{center}
%     %     \Huge Thanks!
%     % \end{center}
%     \printbibliography
%   \end{frame}
% }
