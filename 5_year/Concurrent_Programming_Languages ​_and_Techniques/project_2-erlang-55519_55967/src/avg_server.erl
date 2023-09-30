%%%-------------------------------------------------------------------
%%% @author Asus
%%% @copyright (C) 2022, <COMPANY>
%%% @doc
%%%
%%% @end
%%% Created : 22. nov 2022 14:02
%%%-------------------------------------------------------------------
-module(avg_server).
-author("Asus").

%% API
-export([avg_server/0, avg_server2/1, start_server/0, sum/2, queue_server/1, new_queue/0, push/2, pop/1]).


start_server() ->
  io:format("server running\n"),
  M = spawn(?MODULE, avg_server, []),
  M.

avg_server() ->
  io:format("server running\n"),
  receive
    Num ->
      io:format("server running\n"),
      avg_server2([Num])
  end.
avg_server2(Nums) ->
  io:format("Avg of ~p is ~p ~n", [Nums, sum(0,Nums)/length(Nums)]) ,
  receive
    Num -> avg_server2(Nums ++ [Num])
  end.

sum(Sum, []) -> Sum;
sum(Sum, [H|T]) -> sum(Sum+H,T).

% Main server function
queue_server(Q) ->
  io:format("start server queue\n"),
  receive
    {Pid, push, Val} -> Pid ! ok, queue_server(Q ++ [Val]) ;
    {Pid, pop} when length(Q) > 0 -> Pid ! hd(Q), queue_server(tl(Q))
  end.

% API
new_queue() -> spawn(fun() -> queue_server([]) end).
push(Pid, Val) ->
  Pid ! {self(), push, Val},
  receive ok -> ok end.
pop(Pid) ->
  Pid ! {self(), pop},
  receive Val -> Val end.