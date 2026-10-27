# Structured decompositions of C-sets from tree decompositions

*How to decompose a C-set by tree-decomposing a graph built from it — the way one decomposes a hypergraph through its primal graph.*

---

## Contents

0. Overview
1. Setup
2. Support functions
3. The lifting theorem
4. Examples
5. Completeness: the finest support
6. Computing the decomposition

References

---

## 0. Overview

The standard way to decompose a hypergraph has three steps: form its primal graph, take a tree decomposition of that graph, and cut the hypergraph into the sub-hypergraphs induced on the bags. This note carries out the same construction for an arbitrary C-set $X\colon\mathcal C\to\mathbf{Set}$.

**The data.** All of it is a functor or an adjunction.

- A *support function* is a functor $\sigma\colon(\int X)^{\mathrm{op}}\to\mathcal P(V)$ from the category of elements of $X$ to the power set of a vertex set $V$ (Definition 2.2). Because $\mathrm{Sub}(X)$ is the free sup-lattice on $(\int X)^{\mathrm{op}}$, a support function is the same thing as a join-preserving map $\sigma_*\colon\mathrm{Sub}(X)\to\mathcal P(V)$ (Lemma 2.1).
- The *induced sub-C-sets* are given by the right adjoint $X[-]\colon\mathcal P(V)\to\mathrm{Sub}(X)$ of $\sigma_*$ (Definition 2.3). Concretely, $X[S]=\lbrace x:\sigma(x)\subseteq S\rbrace$.
- The *Gaifman graph* $G_\sigma(X)$ is the 1-skeleton of the simplicial complex spanned by the sets $\sigma(x)$, via the adjunction between 1-skeleta and clique complexes (Definition 2.6).
- A seed that is not yet a functor is completed to one by a left Kan extension, which we call *saturation* (Lemma 2.5).

**The recipe.** Take a tree decomposition $(T,\beta)$ of $G_\sigma(X)$. Then the composite
$$
\textstyle\int T\xrightarrow{\ \hat\beta\ }\mathcal P(V)\xrightarrow{\ X[-]\ }\mathrm{Sub}(X)\subseteq\mathcal C\text{-}\mathbf{Set}
$$
is the lifted decomposition, with bags $X[\beta_t]$ and adhesions $X[\beta_s\cap\beta_t]$.

**What is proved.**

- **Lifting (Theorem 3.2).** The composite is a monic structured decomposition of $X$ if and only if $(T,\beta)$ is a tree decomposition of $G_\sigma(X)$. Equivalently, $X[-]$, which in general preserves only meets, preserves the colimit of $\hat\beta$.
- **Examples (§4).** Different supports recover ordinary, primal and incidence tree decompositions.
- **Completeness (Theorem 5.3).** The finest support is universal: every support factors through it (Proposition 5.1). For the finest support, *every* monic tree-shaped structured decomposition of $X$ arises from the recipe.
- **Computation (§6).** The bags can be extracted in time proportional to their size, starting from one "top" node of the clique tree for each element.

---

## 1. Setup

**C-sets and elements.** Throughout, $\mathcal C$ is a small category and a *C-set* is a functor $X\colon\mathcal C\to\mathbf{Set}$. For $x\in X(c)$ and $f\colon c\to c'$, write $f\cdot x=X(f)(x)$.

The *category of elements* $\int X$ has an object $(c,x)$ for each $x\in X(c)$, and a morphism $f\colon(c,x)\to(c',f\cdot x)$ for each $f\colon c\to c'$. We usually write $x$ for $(c,x)$, and write $\mathrm{El}(X)$ for the set of objects of $\int X$. A C-set $X$ is *finite* if $\mathrm{El}(X)$ is finite.

**Subobjects.** $\mathrm{Sub}(X)$ is the poset of sub-C-sets $Y\subseteq X$, ordered by inclusion. A sub-C-set is a family of subsets $Y(c)\subseteq X(c)$ closed under the action, and we choose these inclusions as representatives of subobjects. $\mathrm{Sub}(X)$ is a complete lattice: joins are unions, and meets are intersections, i.e. pullbacks of monomorphisms. For an element $x\in X(c)$, the sub-C-set $\langle x\rangle$ is the image of the morphism $\mathcal C(c,-)\to X$ that corresponds to $x$ under the Yoneda lemma. It consists of all the elements $f\cdot x$.

Colimits in $\mathcal C\text{-}\mathbf{Set}$ are computed objectwise. A morphism is a monomorphism if and only if it is injective at every object, and an isomorphism if and only if it is bijective at every object.

**Posets.** Posets are regarded as thin categories, so that functors between them are monotone maps. An adjunction $f\dashv g$ between posets is a Galois connection: $f(a)\le b$ if and only if $a\le g(b)$. Left adjoints preserve joins and right adjoints preserve meets. Conversely, a map between complete lattices that preserves all joins has a right adjoint, and a map that preserves all meets has a left adjoint.

**Graphs and complexes.** Graphs are finite, simple and undirected. For a set $V$:

- $\mathbf{Gr}(V)$ is the poset of graphs with vertex set $V$, ordered by inclusion of edge sets.
- $\mathbf{Cx}(V)$ is the poset of simplicial complexes on $V$, meaning families of finite subsets of $V$ closed under taking subsets, ordered by inclusion.

A *clique* is a set of pairwise adjacent vertices; $\varnothing$ and singletons are cliques. $G[W]$ denotes the induced subgraph on $W$.

**Definition 1.1 (tree decomposition).** A *tree decomposition* of a graph $G$ on $V$ is a pair $(T,\beta)$, where $T$ is a finite nonempty tree and $\beta\colon V(T)\to\mathcal P(V)$, satisfying:

- (T1) every vertex lies in some bag $\beta_t$;
- (T2) for every edge $uv$, some bag contains both $u$ and $v$;
- (T3) for every vertex $v$, the set $T_v=\lbrace t:v\in\beta_t\rbrace$ induces a connected subgraph of $T$.

Write $\hat\beta\colon\int T\to\mathcal P(V)$ for the functor $t\mapsto\beta_t$, $st\mapsto\beta_s\cap\beta_t$. Here $\int T$ is as in Definition 1.2.

**Definition 1.2 (structured decomposition).** We follow Bumpus, Kocsis and Master. Regard a graph $H$ (the *shape*) as a C-set on the schema $E\rightrightarrows V$, after choosing an orientation of each edge. Its category of elements $\int H$ has the nodes and edges of $H$ as objects, with one morphism $e\to h$ for each endpoint $h$ of each edge $e$. Up to isomorphism this does not depend on the orientation.

A *structured decomposition of shape $H$* is a functor $d\colon\int H\to\mathcal C\text{-}\mathbf{Set}$. It consists of *bags* $d(h)$ and *adhesion spans* $d(h)\leftarrow d(e)\to d(h')$. We say:

- $d$ *decomposes* $X$ if it comes with a colimit cocone $d\Rightarrow X$;
- $d$ is *monic* if every leg of every adhesion span is a monomorphism;
- $d$ is *tree-shaped* if $H$ is a tree.

**Lemma 1.3 (Helly property for subtrees).** Root the tree $T$. A *subtree* is a nonempty set of nodes that induces a connected subgraph.

1. Every subtree $T'$ has a unique node $\mathrm{top}(T')$ closest to the root, and every node of $T'$ is a descendant of $\mathrm{top}(T')$.
2. Let $T_1,\dots,T_k$ be pairwise intersecting subtrees. Then their tops lie on a single path towards the root, and the deepest of them, $r$, lies in every $T_i$.
3. Under the hypotheses of (2), $\bigcap_iT_i$ is a subtree with top $r$.

*Proof.* (1) Suppose $a\neq b$ are nodes of $T'$ of minimal depth. The path between them passes through their lowest common ancestor, which has smaller depth and lies in $T'$; this is a contradiction. Next, suppose $w\in T'$ is not a descendant of $\mathrm{top}(T')$. Then the path from $w$ to $\mathrm{top}(T')$ passes through a proper ancestor of $\mathrm{top}(T')$, and that ancestor lies in $T'$; again a contradiction.

(2) For $i\neq j$, choose $w\in T_i\cap T_j$. By (1), both tops are ancestors of $w$, so they are comparable, and a set of pairwise comparable nodes lies on a single path towards the root. Let $r=\mathrm{top}(T_j)$ be the deepest top, and fix $i$. Then $r$ lies on the path from $\mathrm{top}(T_i)$ down to $w$, and that path lies in $T_i$.

(3) The unique path in $T$ between two nodes of $\bigcap_iT_i$ lies in every $T_i$, so the intersection is connected. It contains $r$ by (2), and each of its nodes lies in $T_j$, so is a descendant of $r$ by (1). $\square$

---

## 2. Support functions

**Lemma 2.1 (subobjects form the free sup-lattice on elements).** The assignment $x\mapsto\langle x\rangle$ is a functor $\langle-\rangle\colon(\int X)^{\mathrm{op}}\to\mathrm{Sub}(X)$. For every complete lattice $L$ and every functor $F\colon(\int X)^{\mathrm{op}}\to L$, there is a unique join-preserving map $F_*\colon\mathrm{Sub}(X)\to L$ with $F_*\langle x\rangle=F(x)$, namely
$$
F_*(Y)=\bigvee_{y\in\mathrm{El}(Y)}F(y).
$$

*Proof.* $\langle-\rangle$ is a functor because $\langle f\cdot x\rangle\subseteq\langle x\rangle$.

Every sub-C-set is the union of the sub-C-sets generated by its elements, $Y=\bigcup_{y}\langle y\rangle$. So a join-preserving extension must be given by the displayed formula, which proves uniqueness.

The formula preserves joins, because the elements of a union of sub-C-sets are the union of their elements. It extends $F$, because the elements of $\langle x\rangle$ are the $f\cdot x$ and $F(f\cdot x)\le F(x)$. $\square$

Equivalently, $Y\mapsto\mathrm{El}(Y)$ identifies $\mathrm{Sub}(X)$ with the down-closed subsets of the preorder reflection of $(\int X)^{\mathrm{op}}$, and $\langle-\rangle$ with the principal down-sets.

**Definition 2.2 (support function).** Let $V$ be a set, and order its power set $\mathcal P(V)$ by inclusion. A *support function* on $X$ with values in $V$ is a functor
$$
\textstyle\sigma\colon(\int X)^{\mathrm{op}}\to\mathcal P(V).
$$
Concretely, this means $\sigma(f\cdot x)\subseteq\sigma(x)$ for every element $x$ and every morphism $f$ out of its sort. We assume that each $\sigma(x)$ is finite.

By Lemma 2.1, $\sigma$ extends uniquely to a join-preserving map $\sigma_*\colon\mathrm{Sub}(X)\to\mathcal P(V)$, given by $\sigma_*(Y)=\bigcup_{y}\sigma(y)$. We call $\sigma_*(Y)$ the *support* of $Y$.

**Definition 2.3 (induced sub-C-sets).** The map $\sigma_*$ preserves joins between complete lattices, so it has a right adjoint
$$
X[-]\colon\mathcal P(V)\to\mathrm{Sub}(X),\qquad X[S]=\bigcup\lbrace Y:\sigma_*(Y)\subseteq S\rbrace .
$$
We call $X[S]$ the sub-C-set *induced* on $S$.

**Lemma 2.4.**

1. (Adjunction.) $\sigma_*(Y)\subseteq S$ if and only if $Y\subseteq X[S]$. In particular, $\langle x\rangle\subseteq X[S]$ if and only if $\sigma(x)\subseteq S$. Hence $X[S](c)=\lbrace x\in X(c):\sigma(x)\subseteq S\rbrace$.
2. $X[-]$ preserves all meets: $X[\bigcap_iS_i]=\bigcap_iX[S_i]$, and $X[V]=X$.
3. Every meet-preserving map $\mathcal P(V)\to\mathrm{Sub}(X)$ is $X[-]$ for exactly one support function.

*Proof.* (1) The first statement is the definition of the adjunction. For the second, take $Y=\langle x\rangle$ and use $\sigma_*\langle x\rangle=\sigma(x)$. For the third, use that $x\in X[S]$ if and only if $\langle x\rangle\subseteq X[S]$.

(2) Right adjoints preserve meets, and the empty meet is the top element.

(3) A meet-preserving map between complete lattices has a left adjoint. That left adjoint preserves joins, so by Lemma 2.1 it is $\sigma_*$ for exactly one $\sigma$. Finally, adjoints determine each other. $\square$

**Lemma 2.5 (saturation as a Kan extension).** Let $|\int X|$ be the discrete category on $\mathrm{El}(X)$, and let $J\colon|\int X|\to(\int X)^{\mathrm{op}}$ be the inclusion. A *seed* is a functor $\sigma_0\colon|\int X|\to\mathcal P(V)$, that is, an arbitrary map $\mathrm{El}(X)\to\mathcal P(V)$. Its left Kan extension along $J$ is
$$
(\mathrm{Lan}_J\sigma_0)(x)=\bigcup_f\sigma_0(f\cdot x),
$$
where $f$ ranges over the morphisms out of the sort of $x$. This is the smallest support function containing $\sigma_0$: for every support function $\tau$,
$$
\mathrm{Lan}_J\sigma_0\subseteq\tau\quad\text{if and only if}\quad\sigma_0\subseteq\tau J.
$$

*Proof.* The pointwise formula computes the Kan extension at $x$ as the colimit (here, the union) of $\sigma_0$ over the comma category $J\downarrow x$. Its objects are the morphisms $J(y)\to x$ in $(\int X)^{\mathrm{op}}$, that is, the morphisms $x\to y$ in $\int X$, that is, the elements $y=f\cdot x$.

One can also check the claim directly:

- $\sigma(g\cdot x)=\bigcup_f\sigma_0((fg)\cdot x)\subseteq\sigma(x)$, so $\sigma$ is a functor;
- taking $f=\mathrm{id}$ gives $\sigma\supseteq\sigma_0$;
- if $\tau\supseteq\sigma_0$ is a functor, then $\tau(x)\supseteq\tau(f\cdot x)\supseteq\sigma_0(f\cdot x)$. $\square$

In practice, then, one says what the support of each element "obviously" is, and the Kan extension makes it functorial.

**Definition 2.6 (Gaifman graph).** The clique complex $\mathrm{Cl}\colon\mathbf{Gr}(V)\to\mathbf{Cx}(V)$ sends a graph to the set of its finite cliques. It has a left adjoint $\mathrm{sk}_1\colon\mathbf{Cx}(V)\to\mathbf{Gr}(V)$, which takes a complex to its 1-skeleton:
$$
\mathrm{sk}_1K\subseteq G\quad\text{if and only if}\quad K\subseteq\mathrm{Cl}(G).
$$
To see this, suppose first $\mathrm{sk}_1K\subseteq G$, and let $s\in K$. Every 2-element subset of $s$ lies in $K$, hence is an edge of $G$, so $s$ is a clique. Conversely, if $K\subseteq\mathrm{Cl}(G)$, then each 2-element simplex of $K$ is an edge of $G$.

The *support complex* $K_\sigma(X)\in\mathbf{Cx}(V)$ is the smallest complex containing every $\sigma(x)$. The *Gaifman graph* is
$$
G_\sigma(X)=\mathrm{sk}_1K_\sigma(X).
$$
By the adjunction, $G\supseteq G_\sigma(X)$ if and only if every $\sigma(x)$ is a clique of $G$. Concretely, distinct $u,v$ are adjacent if and only if $\lbrace u,v\rbrace\subseteq\sigma(x)$ for some element $x$.

**Remark 2.7 (terminology).** "Support function" is our own name; we do not know a standard term for it in the C-set literature. Two standard ideas underlie it. Taking vertex sorts (Example 4.1) yields exactly the Gaifman graph of a relational structure from model theory (Gaifman 1982). The functoriality condition $\sigma(f\cdot x)\subseteq\sigma(x)$ is the one satisfied by supports in nominal sets (Pitts 2013).

---

## 3. The lifting theorem

**Lemma 3.1 (gluing sub-C-sets).** Let $H$ be a finite graph and $A\colon V(H)\to\mathrm{Sub}(X)$ any assignment. Let $d_A\colon\int H\to\mathcal C\text{-}\mathbf{Set}$ send a node $h$ to $A_h$, and an edge with endpoints $h,h'$ to the meet $A_h\cap A_{h'}$, with inclusions as legs. Let $\iota\colon d_A\Rightarrow X$ be the cocone of inclusions, and for each element $x$ put $H_x=\lbrace h:x\in A_h\rbrace$.

Then $\iota$ is a colimit cocone if and only if, for every element $x$, the set $H_x$ is nonempty and induces a connected subgraph of $H$.

*Proof.* Colimits are computed objectwise, so fix an object $c$. Each element of an adhesion is identified with its copies in the two adjacent bags. So the colimit at $c$ is $\coprod_hA_h(c)$ modulo the equivalence relation generated by $(h,x)\sim(h',x)$, for each edge $hh'$ and each $x\in A_h\cap A_{h'}$.

The comparison map to $X(c)$ sends the class of $(h,x)$ to $x$.

- It is surjective if and only if every $H_x$ is nonempty.
- Each generating identification fixes $x$ and moves along an edge of $H$ inside $H_x$. So the classes lying over $x$ correspond to the connected components of $H[H_x]$, and the map is injective if and only if each $H[H_x]$ is connected.

Finally, $\iota$ is a colimit cocone if and only if the comparison map is bijective at every object. $\square$

Take $\mathcal C=1$, so that C-sets are sets, and $A=\beta$. Then Lemma 3.1 says that (T1) and (T3) together hold exactly when $\hat\beta\colon\int T\to\mathcal P(V)\subseteq\mathbf{Set}$ is a structured decomposition of the set $V$ by its inclusions. So a tree decomposition of $G$ is a monic tree-shaped decomposition of $V$ with intersection adhesions whose bags cover the edges of $G$, which is condition (T2).

**Theorem 3.2 (lifting).** Let $\sigma$ be a support function on $X$ with values in a finite set $V$. Let $T$ be a finite tree, and let $\beta\colon V(T)\to\mathcal P(V)$ satisfy (T1) and (T3). Consider the composite
$$
\textstyle\int T\xrightarrow{\ \hat\beta\ }\mathcal P(V)\xrightarrow{\ X[-]\ }\mathrm{Sub}(X)\subseteq\mathcal C\text{-}\mathbf{Set},
$$
which sends $t\mapsto X[\beta_t]$ and $st\mapsto X[\beta_s\cap\beta_t]=X[\beta_s]\cap X[\beta_t]$. The following are equivalent:

- the composite is a monic structured decomposition of $X$, with the inclusions as colimit cocone;
- every $\sigma(x)$ is contained in some bag;
- $(T,\beta)$ is a tree decomposition of $G_\sigma(X)$.

Consequently, for a graph $G$ on $V$, every tree decomposition of $G$ lifts in this way if and only if $G\supseteq G_\sigma(X)$. In particular, $G_\sigma(X)$ is the smallest graph with this property.

*Proof.* Since $X[-]$ preserves meets (Lemma 2.4(2)), the composite is the diagram $d_A$ of Lemma 3.1 with $A_t=X[\beta_t]$. By Lemma 2.4(1),
$$
T_x:=\lbrace t:x\in X[\beta_t]\rbrace=\lbrace t:\sigma(x)\subseteq\beta_t\rbrace .
$$

*First condition implies second.* If the composite decomposes $X$, then every $T_x$ is nonempty by Lemma 3.1, so $\sigma(x)$ lies in some bag.

*Second condition implies third.* Every edge of $G_\sigma(X)$ lies in some $\sigma(x)$, which gives (T2).

*Third condition implies first.* If $\sigma(x)=\varnothing$, then $T_x=V(T)$, which is connected. Otherwise $T_x=\bigcap_{v\in\sigma(x)}T_v$. These are subtrees by (T1) and (T3), and they pairwise intersect by (T2), since $\sigma(x)$ is a clique of $G_\sigma(X)$. By Lemma 1.3, $T_x$ is a nonempty subtree, and Lemma 3.1 applies.

*The consequence.* If $G\supseteq G_\sigma(X)$, every tree decomposition of $G$ is also one of $G_\sigma(X)$. If $G\not\supseteq G_\sigma(X)$, choose an element $x$ and distinct $u,v\in\sigma(x)$ with $uv\notin E(G)$. Let $T$ have two nodes $1,2$, with $\beta_1=V\setminus\lbrace v\rbrace$ and $\beta_2=V\setminus\lbrace u\rbrace$. This is a tree decomposition of $G$: an edge not containing $v$ lies in $\beta_1$, and an edge containing $v$ cannot contain $u$, so it lies in $\beta_2$. But $\sigma(x)$ lies in neither bag. $\square$

In categorical terms: $\hat\beta$ is a decomposition of the set $V$, with colimit $V$. The right adjoint $X[-]$ preserves meets, but in general not joins or colimits. Theorem 3.2 says that it nevertheless carries this colimit to a colimit, $\operatorname{colim}X[\hat\beta]=X[V]=X$, exactly when the bags cover the supports.

**Proposition 3.3 (local form).** Let $(A,B)$ be a separation of $G_\sigma(X)[W]$: that is, $A\cup B=W$ and no edge joins $A\setminus B$ to $B\setminus A$. Then $X[-]$ preserves this join, $X[A\cup B]=X[A]\cup X[B]$. Moreover, the square of inclusions with corners $X[A\cap B]$, $X[A]$, $X[B]$ and $X[W]$ is a pushout.

*Proof.* Let $x\in X[W]$. Then $\sigma(x)\subseteq W$ is a clique, so it cannot meet both $A\setminus B$ and $B\setminus A$. Hence $\sigma(x)\subseteq A$ or $\sigma(x)\subseteq B$. By Lemma 2.4(2), $X[A]\cap X[B]=X[A\cap B]$. Now apply Lemma 3.1 to the C-set $X[W]$, with the shape consisting of a single edge. $\square$

**Remark 3.4 (sheaves).** Proposition 3.3 is what a sheaf condition for these covers sees. Let $F\colon\mathrm{Sub}(X)^{\mathrm{op}}\to\mathbf{Set}$ be a presheaf, and suppose $F$ sends every square of Proposition 3.3, for every $W$, to a pullback. Then $F(X)$ is the limit of $F$ over every lifted tree decomposition of $X$.

*Sketch.* Cutting an edge $st$ of $T$ splits $T$ into two subtrees. The unions $A$ and $B$ of their bags form a separation of $G_\sigma(X)$ with $A\cap B=\beta_s\cap\beta_t$. A limit over a tree is an iterated pullback, so the claim follows by induction on the number of nodes.

Conversely, suppose $F$ is a sheaf for the lifted tree decompositions of every $X[W]$. Applying this to the two-bag decomposition $(A,B)$ shows that $F$ sends the square to a pullback.

**Remark 3.5 (empty supports and forests).** An element with $\sigma(x)=\varnothing$ lies in every bag, which is why $T$ must be connected. A useful support therefore gives empty support only to elements that genuinely belong everywhere; compare Example 4.4(c).

If $G$ is disconnected, tree-decomposition software often returns a forest. A forest shape is fine as long as no element has empty support. In that case each $\sigma(x)$ is a nonempty clique, so it lies in one component of $G$, and $T_x$ lies inside that component's tree. Otherwise, first join the component trees into a single tree; this preserves (T1)–(T3) (§6.6).

---

## 4. Examples

**4.1 Vertex-sort supports.** Choose a set $V_0$ of objects of $\mathcal C$, the *vertex sorts*, and let $V=\coprod_{v\in V_0}X(v)\subseteq\mathrm{El}(X)$. Set
$$
\sigma(x)=\mathrm{El}\langle x\rangle\cap V=\lbrace f\cdot x:\ f\colon c\to v,\ v\in V_0\rbrace .
$$
This support is the finest support of §5 followed by intersection with $V$. Equivalently, it is the Kan extension of the seed that sends $x\mapsto\lbrace x\rbrace$ on vertex sorts and $x\mapsto\varnothing$ elsewhere. Elements of the other sorts behave like tuples: their support is whatever they point to.

**4.2 Graphs.** The schema has two morphisms $s,t\colon E\to V$, and we take $V_0=\lbrace V\rbrace$. Then $\sigma(v)=\lbrace v\rbrace$ and $\sigma(e)=\lbrace s\cdot e,\,t\cdot e\rbrace$. The Gaifman graph $G_\sigma(X)$ is the underlying simple graph, with loops and parallel edges forgotten, and $X[S]$ is the induced subgraph on $S$. In this case Theorem 3.2 is the usual reading of a tree decomposition of a graph as a structured decomposition.

For symmetric graphs, add the involution $\iota\colon E\to E$ with $s\iota=t$. Nothing changes, because $\sigma(\iota\cdot e)=\sigma(e)$.

**4.3 Semi-simplicial and simplicial sets.** Take $\mathcal C=\Delta_{\mathrm{inj}}^{\mathrm{op}}$ (or $\Delta^{\mathrm{op}}$) and $V_0=\lbrace[0]\rbrace$. Then $\sigma(x)$ is the set of vertices of the simplex $x$. The Gaifman graph $G_\sigma(X)$ is the underlying simple graph of the 1-skeleton, and $X[S]$ is the full sub-(semi-)simplicial set on the vertices in $S$.

**4.4 Hypergraphs.** Use the incidence schema: objects $V,E,I$ and morphisms $\nu\colon I\to V$ and $\varepsilon\colon I\to E$. An element $i\in X(I)$ is an incidence between the vertex $\nu\cdot i$ and the hyperedge $\varepsilon\cdot i$. Write $N(e)=\lbrace\nu\cdot i:\varepsilon\cdot i=e\rbrace$ for the vertex set of a hyperedge $e$.

- **(a) Incidence.** Take $V_0=\lbrace V,E\rbrace$. Then $\sigma(v)=\lbrace v\rbrace$, $\sigma(e)=\lbrace e\rbrace$ and $\sigma(i)=\lbrace\nu\cdot i,\,\varepsilon\cdot i\rbrace$. $G_\sigma(X)$ is the bipartite incidence graph, and $X[S]$ keeps the vertices and hyperedges in $S$ together with the incidences between them. This gives *incidence treewidth*.
- **(b) Primal.** Take $V=X(V)$, with $\sigma(v)=\lbrace v\rbrace$, $\sigma(e)=N(e)$ and $\sigma(i)=N(\varepsilon\cdot i)$. This is the Kan extension of the seed $\sigma_0(v)=\lbrace v\rbrace$, $\sigma_0(e)=N(e)$, $\sigma_0(i)=\varnothing$. $G_\sigma(X)$ is the primal graph, and $X[S]$ is the induced sub-hypergraph: the vertices in $S$, the hyperedges $e$ with $N(e)\subseteq S$, and their incidences. This gives *primal treewidth*, and it is the construction from the overview.
- **(c) Vertex sort only.** Take $V_0=\lbrace V\rbrace$. There is no morphism $E\to V$, so $\sigma(e)=\varnothing$, while $\sigma(i)=\lbrace\nu\cdot i\rbrace$. $G_\sigma(X)$ has no edges, and every hyperedge lies in every bag. The resulting decompositions are valid but useless.

The contrast between (b) and (c) is the moral. The right support need not have the vertex-sort form $\mathrm{El}\langle x\rangle\cap V$: in (b), the support of a hyperedge reaches *backwards* along $\varepsilon$.

**4.5 Seeds given by spans.** All of the examples above can be specified in the schema. For each object $c$, list spans of morphisms $c\xleftarrow{\,p\,}d\xrightarrow{\,q\,}v$ with $v\in V_0$. For $x\in X(c)$, set
$$
\sigma_0(x)=X(q)\bigl(X(p)^{-1}(x)\bigr),
$$
that is, pull back along $p$ and push forward along $q$, taking the union over the listed spans. Then take the Kan extension. A forward path to a vertex sort is the case $p=\mathrm{id}$. For example, the primal support (b) uses the span $E\xleftarrow{\,\varepsilon\,}I\xrightarrow{\,\nu\,}V$ for hyperedges and the identity span on $V$ for vertices.

| C-set | Support | Gaifman graph | Recovers |
|---|---|---|---|
| Graph | endpoints (4.2) | underlying simple graph | ordinary tree decompositions |
| (Semi-)simplicial set | vertices of a simplex (4.3) | 1-skeleton | — |
| Hypergraph | 4.4(a) | incidence graph | incidence treewidth |
| Hypergraph | 4.4(b) | primal graph | primal treewidth |

---

## 5. Completeness: the finest support

In this section $X$ is finite. The *finest support* is the composite
$$
\textstyle\sigma_\downarrow\colon(\int X)^{\mathrm{op}}\xrightarrow{\ \langle-\rangle\ }\mathrm{Sub}(X)\xrightarrow{\ \mathrm{El}\ }\mathcal P(\mathrm{El}(X)),\qquad\sigma_\downarrow(x)=\mathrm{El}\langle x\rangle .
$$
Its extension $(\sigma_\downarrow)_*$ is the inclusion $\mathrm{El}\colon\mathrm{Sub}(X)\hookrightarrow\mathcal P(\mathrm{El}(X))$. Its right adjoint $X_\downarrow[-]$ is the interior operator: $X_\downarrow[S]$ is the largest sub-C-set whose elements all lie in $S$. In the Gaifman graph $G_\downarrow(X):=G_{\sigma_\downarrow}(X)$, two elements are adjacent if and only if both lie in $\langle z\rangle$ for some element $z$.

**Proposition 5.1 (universality).** Let $\sigma$ be any support function with values in $V$. Let $s_!\colon\mathcal P(\mathrm{El}(X))\to\mathcal P(V)$ be the join-preserving map $S\mapsto\bigcup_{y\in S}\sigma(y)$, and let $s^*$ be its right adjoint, $s^*(S)=\lbrace y:\sigma(y)\subseteq S\rbrace$. Then
$$
\sigma_*=s_!\circ\mathrm{El},\qquad X[-]=X_\downarrow[-]\circ s^*,
$$
and in fact $X[S]$ is the sub-C-set with elements $s^*(S)$. Consequently, the $\sigma$-lift of $(T,\beta)$ is the $\sigma_\downarrow$-lift of $(T,s^*\beta)$.

*Proof.* Both $\sigma_*$ and $s_!\circ\mathrm{El}$ preserve joins. They agree on each $\langle x\rangle$, since $s_!(\mathrm{El}\langle x\rangle)=\bigcup_f\sigma(f\cdot x)=\sigma(x)$. By Lemma 2.1 they are therefore equal.

The right adjoint of a composite is the composite of the right adjoints, which gives the second equation. The set $s^*(S)$ is already down-closed, because $\sigma$ is a functor, so the interior does nothing. Finally, $s^*$ preserves intersections, so $s^*(\beta_s\cap\beta_t)=s^*\beta_s\cap s^*\beta_t$. $\square$

So every support factors through the finest one, and every $\sigma$-lifted decomposition is also a $\sigma_\downarrow$-lifted decomposition. The converse — that the $\sigma_\downarrow$-lifts already exhaust all monic tree-shaped decompositions — rests on the following lemma about $\mathbf{Set}$.

**Lemma 5.2 (tree-shaped diagrams of injections).** Let $T$ be a finite tree, and let $D\colon\int T\to\mathbf{Set}$ be a diagram with injective legs and colimit cocone $\lambda\colon D\Rightarrow L$. Then:

1. every $\lambda_t$ is injective;
2. for every edge $st$, $\lambda_s(D(s))\cap\lambda_t(D(t))=\lambda_{st}(D(st))$;
3. for every $y\in L$, the set $\lbrace t:y\in\lambda_t(D(t))\rbrace$ is a subtree of $T$.

*Proof.* For an edge $e=st$, write $\iota_s,\iota_t$ for its legs. Let $\Gamma$ be the graph whose vertices are the pairs $(t,a)$ with $a\in D(t)$, with one edge joining $(s,\iota_sc)$ and $(t,\iota_tc)$ for each edge $e=st$ of $T$ and each $c\in D(e)$. Every element of an adhesion is identified with an element of a bag, so $L$ is the set of connected components of $\Gamma$, and $\lambda_t(a)$ is the component containing $(t,a)$.

The projection $p\colon\Gamma\to T$, $(t,a)\mapsto t$, is a graph homomorphism. It is also *locally injective*: at a vertex $(s,a)$, at most one edge lies over each tree edge $st$, because $\iota_s$ is injective.

*Claim: $p$ is injective on each connected component of $\Gamma$.* Otherwise, choose distinct $u,w$ in one component with $p(u)=p(w)$, and take a shortest path from $u$ to $w$. It has length at least 1, and consecutive edges of it are distinct. By local injectivity, their images are distinct tree edges. So the image is a closed walk of positive length in $T$ that never immediately retraces an edge. A tree has no such walk.

We can now prove the three statements.

- (1) Distinct elements $a,a'\in D(t)$ have the same image $t$, so by the claim they lie in different components.
- (3) The image of a connected component is connected.
- (2) Suppose $y=\lambda_s(a)=\lambda_t(b)$, where $st$ is an edge of $T$, and let $\Gamma_y$ be the component $y$. By the claim, $p$ restricts to an injective homomorphism on $\Gamma_y$. Its image is a connected subgraph of the tree $T[p(\Gamma_y)]$ that contains every node of it, so it is all of $T[p(\Gamma_y)]$, and in particular it contains the edge $st$. By injectivity of $p$ on $\Gamma_y$, the edge of $\Gamma_y$ lying over $st$ joins $(s,a)$ and $(t,b)$. So $a=\iota_sc$ and $b=\iota_tc$ for some $c\in D(st)$, and $y=\lambda_{st}(c)$. The reverse inclusion is clear. $\square$

**Theorem 5.3 (completeness).** Let $T$ be a finite tree, and let $d\colon\int T\to\mathcal C\text{-}\mathbf{Set}$ be a monic structured decomposition of a finite C-set $X$, with colimit cocone $\lambda$. Then:

1. each $\lambda_t$ is a monomorphism; write $A_t\in\mathrm{Sub}(X)$ for its image;
2. for each edge $st$, $\lambda_{st}$ maps $d(st)$ isomorphically onto $A_s\cap A_t$;
3. setting $\beta_t=\mathrm{El}(A_t)$ defines a tree decomposition $(T,\beta)$ of $G_\downarrow(X)$, with $X_\downarrow[\beta_t]=A_t$.

Consequently, $d$ is isomorphic, as a diagram over $X$, to the $\sigma_\downarrow$-lift of $(T,\beta)$.

*Proof.* Colimits are computed objectwise. Evaluating at each object therefore gives a tree-shaped diagram of injections in $\mathbf{Set}$, and Lemma 5.2 gives (1) and (2). Each $A_t$ is a sub-C-set because $\lambda_t$ is natural.

For (3), we check the axioms of a tree decomposition.

- (T1) is the nonemptiness in Lemma 5.2(3).
- For (T2), suppose $x,y\in\langle z\rangle$. Pick $t$ with $z\in A_t$. Then $\langle z\rangle\subseteq A_t$, so $x,y\in\beta_t$.
- (T3) is Lemma 5.2(3), applied at the sort of each element.

The equation $X_\downarrow[\mathrm{El}(A_t)]=A_t$ holds because $X_\downarrow[-]$ is the interior operator and $A_t$ is already a sub-C-set. By meet preservation, the adhesions of the lift are then $A_s\cap A_t$. The isomorphisms $\lambda_t$ and $\lambda_{st}$ commute with the legs and with the cocones. $\square$

**Corollary 5.4.** Let $X$ be a finite C-set and $T$ a finite tree. Lifting along $\sigma_\downarrow$ is a bijection between:

- tree decompositions $(T,\beta)$ of $G_\downarrow(X)$ whose bags are *downward closed*, meaning each $\beta_t$ lies in the image of $\mathrm{El}\colon\mathrm{Sub}(X)\hookrightarrow\mathcal P(\mathrm{El}(X))$; and
- monic structured decompositions of $X$ of shape $T$, up to isomorphism of diagrams over $X$.

Moreover, any tree decomposition $(T,\beta)$ of $G_\downarrow(X)$ can be replaced by the downward-closed decomposition $\beta'_t=\mathrm{El}(X_\downarrow[\beta_t])$ without changing its lift.

*Proof.* Theorem 5.3 provides the inverse. Starting from a downward-closed $\beta$, the lift has images $X_\downarrow[\beta_t]$, whose sets of elements are exactly the $\beta_t$.

For the last sentence, apply Theorem 5.3 to the lift of $\beta$. Its images are $X_\downarrow[\beta_t]$, so $\beta'$ is a tree decomposition. Moreover $X_\downarrow[\beta'_t]=X_\downarrow[\beta_t]$ because $X_\downarrow[-]\circ\mathrm{El}\circ X_\downarrow[-]=X_\downarrow[-]$; the adhesions agree by meet preservation. $\square$

So for C-sets, "decompose a graph, then lift" loses nothing: every monic tree-shaped structured decomposition is a lifted tree decomposition of $G_\downarrow(X)$.

**Remark 5.5 (coarse versus fine).** By Proposition 5.1, a support $\sigma$ selects the subfamily of the decompositions in Corollary 5.4 whose bags have the form $s^*(S)$. A coarser support gives a smaller graph and fewer decompositions.

For example, let $X$ be a graph. Then $G_\downarrow(X)$ is the underlying simple graph together with one extra vertex for each edge, adjacent to that edge's endpoints. Suppose $X$ has a non-loop edge, and its underlying simple graph has treewidth $k$. Then $\mathrm{tw}\,G_\downarrow(X)=\max(k,2)$:

- for the upper bound, attach a bag $\lbrace e,\,s\cdot e,\,t\cdot e\rbrace$ to a bag containing $s\cdot e$ and $t\cdot e$;
- for the lower bound, the underlying simple graph is an induced subgraph, and each non-loop edge contributes a triangle.

---

## 6. Computing the decomposition

The input is a finite C-set $X$ together with a seed, for example one specified by spans as in 4.5.

**6.1 Saturate.** Present $\mathcal C$ by generating morphisms. Every morphism is a composite of generators, so the Kan extension $\sigma=\mathrm{Lan}_J\sigma_0$ is the least function satisfying
$$
\sigma(x)=\sigma_0(x)\cup\bigcup_g\sigma(g\cdot x),
$$
where $g$ ranges over the generators out of the sort of $x$. If the graph of generators is acyclic, compute $\sigma$ sort by sort, starting with the sorts that have no outgoing generators. Otherwise, iterate to a fixed point.

**6.2 Build the Gaifman graph.** Make each $\sigma(x)$ a clique. This costs $\sum_x\lvert\sigma(x)\rvert^2$; elements with $\lvert\sigma(x)\rvert\le1$ contribute nothing.

**6.3 Tree-decompose.** Any tree-decomposition algorithm will do. Clique trees computed from an elimination ordering, for example `perm, tree = cliquetree(G)` in CliqueTrees.jl, have the following structure, which is all we use:

- each node $t$ has a bag $\beta_t=\mathrm{res}(t)\sqcup\mathrm{sep}(t)$;
- the residuals partition $V$, and the vertices in $\mathrm{res}(t)$ are the ones eliminated at $t$;
- $\mathrm{sep}(t)\subseteq\beta_{\mathrm{parent}(t)}$;
- a vertex in $\mathrm{res}(t)$ occurs only in the bags of $t$ and its descendants.

Consequently, $\beta_c\cap\beta_{\mathrm{parent}(c)}=\mathrm{sep}(c)$ for every non-root node $c$, and the top of $T_v$ is the node whose residual contains $v$.

**6.4 Top nodes.** For an element $x$ with $\sigma(x)\neq\varnothing$, let $\mathrm{top}(x)$ be the node whose residual contains the vertex of $\sigma(x)$ that is *eliminated first*. In CliqueTrees.jl, this is the vertex with the smallest permuted label. Then $\mathrm{top}(x)$ is the top of $T_x=\lbrace t:\sigma(x)\subseteq\beta_t\rbrace$.

Indeed, $T_x=\bigcap_{v\in\sigma(x)}T_v$, so by Lemma 1.3 its top is the deepest of the nodes $\mathrm{top}(T_v)$. Now let $u,v\in\sigma(x)$, with $u$ eliminated before $v$. They are adjacent, so $v$ lies in the bag where $u$ is eliminated. That node is therefore in $T_v$, hence a descendant of the node where $v$ is eliminated.

**6.5 Bags and adhesions by descent.** $T_x$ is a subtree with top $\mathrm{top}(x)$, so it can be enumerated by depth-first search from $\mathrm{top}(x)$. By 6.3, the search should enter a child $c$ of a node in $T_x$ exactly when $\sigma(x)\subseteq\mathrm{sep}(c)$. This gives
$$
X[\beta_t]=\lbrace x: t\in T_x\rbrace,\qquad X[\mathrm{sep}(t)]=\lbrace x: t\in T_x,\ t\neq\mathrm{top}(x)\rbrace,
$$
which are the bag at $t$ and the adhesion between $t$ and its parent. Each element costs $\lvert\sigma(x)\rvert$ per child examined. Summed over all elements, this is essentially the size of the output.

**6.6 Empty supports and forests.** If $\sigma(x)=\varnothing$, then $T_x$ is the whole tree, and $\mathrm{top}(x)$ is the root. If $G_\sigma(X)$ is disconnected, `cliquetree` returns a forest. If, in addition, some element has empty support, first make one root the parent of the others (Remark 3.5). The former roots have empty separators, so the result still has the structure of 6.3.

**6.7 Use with the sheaf algorithm.** Each element $x$ has exactly one node, $\mathrm{top}(x)$, at which it lies in the bag but not in the adhesion to the parent. A bag-by-bag dynamic program, such as the sheaf-deciding algorithm of Althaus, Bumpus, Fairbanks and Rosiak, should check or weight $x$ at this node, so that $x$ is counted exactly once.

**Cost.** In total: saturation, plus $O\bigl(\sum_x\lvert\sigma(x)\rvert^2\bigr)$ for the graph, plus the tree decomposition itself, plus output-sensitive work for the bags.

---

## References

- E. Althaus, B. M. Bumpus, J. Fairbanks, D. Rosiak. Compositional algorithms on compositional data: deciding sheaves on presheaves. arXiv:2302.05575 (2023).
- B. M. Bumpus, Z. A. Kocsis, J. E. Master (and E. Minichiello in later versions). Structured decompositions: structural and algorithmic compositionality. arXiv:2207.06091 (2022).
- H. Gaifman. On local and non-local properties. In *Proceedings of the Herbrand Symposium (Logic Colloquium '81)*, North-Holland (1982).
- F. Gavril. The intersection graphs of subtrees in trees are exactly the chordal graphs. *J. Combin. Theory Ser. B* 16 (1974).
- A. M. Pitts. *Nominal Sets: Names and Symmetry in Computer Science*. Cambridge University Press (2013).
- CliqueTrees.jl. https://github.com/AlgebraicJulia/CliqueTrees.jl
