# SQL Fundamentals for Data Analysis

## What is SQL?

SQL (Structured Query Language) is the standard language for interacting with relational
databases. It is used to query, insert, update, and delete data stored in tables.

## SELECT Statement

The SELECT statement retrieves data from one or more tables. A basic query looks like:
`SELECT column1, column2 FROM table_name WHERE condition;`
The WHERE clause filters rows, ORDER BY sorts the result set, and LIMIT restricts the number of
rows returned.

## JOINs

JOINs combine rows from two or more tables based on a related column.
- INNER JOIN returns only rows that have matching values in both tables.
- LEFT JOIN returns all rows from the left table and matched rows from the right table, with NULLs
  where there is no match.
- RIGHT JOIN is the mirror of LEFT JOIN.
- FULL OUTER JOIN returns all rows when there is a match in either table.

## GROUP BY and Aggregate Functions

GROUP BY groups rows that share a value in specified columns so aggregate functions (COUNT, SUM,
AVG, MIN, MAX) can be applied per group. The HAVING clause filters groups after aggregation, in
contrast to WHERE, which filters individual rows before aggregation.

## Primary Keys and Foreign Keys

A primary key uniquely identifies each row in a table and cannot contain NULL values. A foreign
key is a column (or set of columns) in one table that references the primary key of another table,
enforcing referential integrity between the two tables.

## Indexes

An index is a data structure that improves the speed of data retrieval operations on a table, at
the cost of additional storage and slower write operations (since the index must be updated on
every insert/update/delete). Indexes are commonly created on columns used frequently in WHERE
clauses or JOIN conditions.

## Normalization

Normalization is the process of organizing database tables to reduce data redundancy and improve
data integrity. Common normal forms include First Normal Form (1NF - atomic columns, no repeating
groups), Second Normal Form (2NF - no partial dependency on a composite key), and Third Normal Form
(3NF - no transitive dependency on non-key columns).

## Window Functions

Window functions perform calculations across a set of rows related to the current row without
collapsing the result into a single row, unlike GROUP BY. Common window functions include
ROW_NUMBER(), RANK(), DENSE_RANK(), and aggregate functions used with an OVER() clause, e.g.
`SUM(sales) OVER (PARTITION BY region ORDER BY date)`.

## Subqueries vs CTEs

A subquery is a query nested inside another query. A Common Table Expression (CTE), defined with
`WITH name AS (...)`, is a named temporary result set that can make complex queries with multiple
subqueries easier to read and can be referenced multiple times within the same statement.
